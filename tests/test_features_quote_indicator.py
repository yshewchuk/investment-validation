"""``has_implied_quote`` is always boolean or absent -- never NaN.

Supports the classification of the STR-THRU strict-capture gap
("feature abs_move.has_implied_quote is missing or nonnumeric",
tools/capture_tier0_corpus.py's `_coerce_feature_value`) as a legitimate
typed gap, not a collector bug: `add_quote_indicators` (engine/features.py)
either produces a real 0.0/1.0 float from `or_implied`, or -- when
`or_implied` itself never landed in the frame -- leaves the column out
entirely. There is no third state where the column exists but is null, so a
captured `None` for this feature can only mean the whole market block never
reached `built` for that row, which is exactly what legacy's own
`missing = [f for f in artifact.features if f not in features.columns]`
check (engine/score.py `_score_model`) also treats as MISSING_FEATURES.

No data/ or panel access: this is a pure-pandas unit test of one function.
"""
from __future__ import annotations

import pandas as pd

from engine.features import add_quote_indicators


def test_has_implied_quote_is_zero_when_the_source_is_zero_or_nan():
    frame = pd.DataFrame({"or_implied": [0.0, 5.0, float("nan")]})

    out = add_quote_indicators(frame)

    assert out["has_implied_quote"].tolist() == [0.0, 1.0, 0.0]
    assert out["has_implied_quote"].notna().all()


def test_has_implied_quote_is_absent_not_null_when_the_source_column_is_missing():
    frame = pd.DataFrame({"other": [1, 2, 3]})

    out = add_quote_indicators(frame)

    assert "has_implied_quote" not in out.columns
