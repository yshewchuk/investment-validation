"""Synthetic checks for the distinct-knot guard and the spline step shape."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import SplineTransformer

from engine.v2.data.errors import DataError
from engine.v2.models.training._magnitude_spline_step import distinct_knots, spline_step


def _continuous(rows=200, columns=3, seed=0):
    return np.random.default_rng(seed).normal(size=(rows, columns))


def test_distinct_knots_counts_quantile_values():
    continuous = _continuous(rows=200, columns=1, seed=7)[:, 0]
    assert distinct_knots(continuous, 6) == 6

    tied = continuous.copy()
    tied[:80] = 0.0
    assert distinct_knots(tied, 6) < 6


def test_spline_step_keeps_raw_columns_then_appends_basis():
    data = _continuous()
    transformer = spline_step(("a", "b", "c"), ("a", "c"), 6)

    out = transformer.fit_transform(data)

    assert out.shape == (200, 3 + 2 * (6 + 1))
    assert np.array_equal(out[:, :3], data)
    assert np.isfinite(out).all()


def test_spline_step_uses_training_quantile_knots_and_appends_in_column_order():
    data = np.random.default_rng(0).normal(size=(200, 3))
    transformer = spline_step(("a", "b", "c"), ("a", "c"), 6)

    out = transformer.fit_transform(data)

    quantiles = 100 * np.linspace(0, 1, 6)
    spline = transformer.named_transformers_["spline"]
    np.testing.assert_allclose(
        spline.bsplines_[0].t[3:-3], np.percentile(data[:, 0], quantiles))
    np.testing.assert_allclose(
        spline.bsplines_[1].t[3:-3], np.percentile(data[:, 2], quantiles))

    expected = SplineTransformer(
        n_knots=6, degree=3, knots="quantile", extrapolation="constant",
        include_bias=False).fit_transform(data[:, [0, 2]])
    np.testing.assert_allclose(out[:, 3:], expected)
    assert (out[:, 3:] >= 0.0).all() and (out[:, 3:] <= 1.0).all()


def test_tied_column_refuses_the_variant():
    data = _continuous(seed=11)
    data[:80, 1] = 0.0

    with pytest.raises(DataError) as raised:
        spline_step(("a", "b", "c"), ("b",), 6).fit(data)

    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"


@pytest.mark.parametrize("fill", [0.0, 5.0])
def test_constant_or_all_zero_column_refuses_the_variant(fill):
    data = np.zeros((50, 2))
    data[:, 1] = fill

    with pytest.raises(DataError) as raised:
        spline_step(("a", "b"), ("b",), 6).fit(data)

    assert raised.value.code == "EXPERIMENT_VARIANT_FAILED"


def test_evaluation_rows_extrapolate_finitely_and_keep_training_knots():
    transformer = spline_step(("a", "b"), ("a",), 6)
    transformer.fit(_continuous(columns=2, seed=3) + 10.0)
    spline = transformer.named_transformers_["spline"]
    before = [bspline.t.copy() for bspline in spline.bsplines_]

    out = transformer.transform(np.array([[-100.0, 1.0], [100.0, 2.0]]))

    assert np.isfinite(out).all()
    for old, new in zip(before, (bspline.t for bspline in spline.bsplines_)):
        assert np.array_equal(old, new)


def test_unknown_spline_column_is_a_contract_mismatch():
    with pytest.raises(DataError) as raised:
        spline_step(("a", "b"), ("c",), 6)

    assert raised.value.code == "CONTRACT_MISMATCH"


@pytest.mark.parametrize("args", [
    (("a", "b"), ("a",), 1),
    (("a", "b"), ("a",), 0),
    (("a", "b"), ("a",), True),
    (("a", "b"), ("a",), 2.5),
    (("a", "b"), ("a", "a"), 6),
])
def test_invalid_spline_arguments_are_contract_mismatches(args):
    with pytest.raises(DataError) as raised:
        spline_step(*args)

    assert raised.value.code == "CONTRACT_MISMATCH"
