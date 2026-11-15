# packages: engine.v2.models.training
"""Synthetic checks for the read-only spline tie measurement helpers."""
from __future__ import annotations

import numpy as np

from tools import spline_tie_measure


def test_year_of_reads_the_event_id_suffix():
    assert spline_tie_measure.year_of("AAL_2014-01-28") == 2014


def _matrix():
    years = np.array([2010 + index % 5 for index in range(50)])
    matrix = np.random.default_rng(0).normal(size=(50, 2))
    matrix[:20, 1] = 0.0
    matrix[20:25, 1] = np.nan
    return matrix, years


def test_measure_reports_only_the_tied_column():
    matrix, years = _matrix()

    result = spline_tie_measure.measure(
        matrix, years, ("continuous", "tied"), (2014,))

    assert result["tied"]["tied_years"][6] == [2014]
    assert result["tied"]["max_share"] >= 0.4
    assert result["continuous"]["tied_years"][6] == []
    assert result["continuous"]["max_share"] < 0.4


def test_format_table_lists_only_columns_tied_at_some_k():
    matrix, years = _matrix()
    result = spline_tie_measure.measure(
        matrix, years, ("continuous", "tied"), (2014,))

    text = spline_tie_measure.format_table(result, (4, 6, 8))

    assert "continuous" in text and "tied" in text
    final = text.strip().splitlines()[-1]
    assert final.startswith("tied columns at any k:")
    assert "tied" in final and "continuous" not in final
