"""Row-identity tests for the computed-moves pure math (issue #41).

``build_rows``/``_row`` stamps every row with a real wall-clock
``computed_at``. A same-session rerun over identical inputs must not look
like new content just because wall-clock time moved between the two calls --
``canonical_row``/``row_content_hash`` are the module's answer, mirroring
``engine.v2.scoring.identity``'s "pop the operational field before hashing"
convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.v2.data.computed_moves import build_rows, canonical_row, row_content_hash


def _inputs():
    sd = pd.to_datetime([
        "2026-01-01", "2026-01-04",
        "2026-02-01", "2026-02-04",
        "2026-03-01", "2026-03-04",
        "2026-04-01", "2026-04-04",
        "2026-05-01", "2026-05-04",
    ]).to_numpy()
    sc = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                   106.0, 107.0, 108.0, 109.0])
    daily = pd.DataFrame({"date": sd, "implied_move": np.linspace(3.0, 4.0, len(sd))})
    events = pd.DataFrame({
        "event_date": pd.to_datetime(
            ["2026-01-02", "2026-02-02", "2026-03-02", "2026-04-02", "2026-05-02"]),
        "session": ["BMO"] * 5,
    })
    return events, sd, sc, daily


def _rows(computed_at: str) -> list[dict]:
    events, sd, sc, daily = _inputs()
    return build_rows(
        "AAPL", events, sd, sc, daily,
        computed_at=computed_at, source_hash="sha256:" + "b" * 64,
        capture_id="capture_fixed")


def test_a_same_session_rerun_gives_an_identical_row_content_hash():
    """The exact defect issue #41 reports: only wall-clock time differs
    between the two calls (as a real rerun would), everything else -- the
    events, prices, source_hash, capture_id -- is identical."""
    first = _rows("2026-09-26T10:00:00+00:00")
    second = _rows("2026-09-26T10:00:07+00:00")

    assert len(first) == 5
    assert first != second, "sanity: the raw rows do differ (computed_at moved)"
    assert [canonical_row(row) for row in first] == [canonical_row(row) for row in second]
    assert [row_content_hash(row) for row in first] == [row_content_hash(row) for row in second]


def test_canonical_row_excludes_computed_at_but_nothing_else():
    row = _rows("2026-09-26T10:00:00+00:00")[0]
    canonical = canonical_row(row)

    assert "computed_at" not in canonical
    assert set(row) - set(canonical) == {"computed_at"}
    for key in canonical:
        assert canonical[key] == row[key]