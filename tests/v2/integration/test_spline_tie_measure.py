# packages: engine.v2.models.training
"""Synthetic checks for the read-only spline tie measurement helpers."""
from __future__ import annotations

import numpy as np
import pytest

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
    assert final == "tied columns at any k: tied"


def test_measure_rejects_a_test_year_with_no_training_rows():
    matrix, years = _matrix()

    with pytest.raises(ValueError, match="no training rows before 2010"):
        spline_tie_measure.measure(
            matrix, years, ("continuous", "tied"), (2010,))


def test_read_request_rejects_a_directory(tmp_path):
    with pytest.raises(SystemExit, match="regular file"):
        spline_tie_measure._read_request(tmp_path)


def test_read_request_rejects_a_file_over_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(spline_tie_measure, "_MAX_REQUEST_BYTES", 10)
    request = tmp_path / "request.json"
    request.write_bytes(b"x" * 20)

    with pytest.raises(SystemExit, match="exceeds 64 MiB"):
        spline_tie_measure._read_request(request)


def test_read_request_parses_a_small_file(tmp_path):
    request = tmp_path / "request.json"
    request.write_text('{"price_history_snapshot": "abc"}', encoding="utf-8")

    assert spline_tie_measure._read_request(request) == {
        "price_history_snapshot": "abc"}


def _horizon_rows():
    rows = []
    index = 0
    next_positive = 1.0
    next_continuous = 1000.0
    for year in range(2010, 2015):
        for slot in range(5):
            if slot < 2:
                tied_value = 0.0
            else:
                tied_value = next_positive
                next_positive += 1.0
            rows.append({"event_id": f"T{index}_{year}-01-05",
                         "horizon_sessions": 1, "col": tied_value})
            index += 1
            rows.append({"event_id": f"T{index}_{year}-01-05",
                         "horizon_sessions": 3, "col": next_continuous})
            next_continuous += 1.0
            index += 1
    return rows


def test_split_by_horizon_and_measure_keep_horizons_separate():
    rows = _horizon_rows()

    groups = spline_tie_measure.split_by_horizon(rows, ("col",))

    assert set(groups) == {1, 3}
    assert groups[1][0].shape == (25, 1) and len(groups[1][1]) == 25
    assert groups[3][0].shape == (25, 1) and len(groups[3][1]) == 25

    horizon_one = spline_tie_measure.measure(*groups[1], ("col",), (2014,))
    horizon_three = spline_tie_measure.measure(*groups[3], ("col",), (2014,))
    assert horizon_one["col"]["tied_years"][6] == [2014]
    assert horizon_three["col"]["tied_years"][6] == []

    pooled_matrix = np.vstack([groups[1][0], groups[3][0]])
    pooled_years = np.concatenate([groups[1][1], groups[3][1]])
    pooled = spline_tie_measure.measure(
        pooled_matrix, pooled_years, ("col",), (2014,))
    assert pooled["col"]["tied_years"][6] == []


def test_split_by_horizon_rejects_a_row_without_horizon_sessions():
    rows = [{"event_id": "T0_2010-01-05", "col": 1.0}]

    with pytest.raises(ValueError, match="row without horizon_sessions"):
        spline_tie_measure.split_by_horizon(rows, ("col",))


def test_measure_computes_exact_values_and_excludes_the_test_year():
    values = [0.0] * 5 + [1.0, 2.0, 3.0, 4.0, 5.0]
    matrix = np.array([[v] for v in values] + [[100.0 + i] for i in range(10)])
    years = np.array([2010] * 10 + [2011] * 10)

    result = spline_tie_measure.measure(matrix, years, ("col",), (2011,), ks=(6,))

    # Training rows are 2010 only (the test year 2011 rows must not leak in):
    # share 5/10; k=6 quantile knots 0, 0, 0, 1.4, 3.2, 5 -> 4 distinct.
    assert result["col"]["max_share"] == 0.5
    assert result["col"]["worst_year"] == 2011
    assert result["col"]["min_distinct"] == {6: 4}
    assert result["col"]["tied_years"] == {6: [2011]}
