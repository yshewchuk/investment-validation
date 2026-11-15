from __future__ import annotations

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.research.derived_features import (
    LOG_COLUMNS,
    PRODUCT_BASE,
    PRODUCT_PAIRS,
    SET_NAMES,
    derive_columns,
    derived_names,
)


def _sources(rows: int = 3) -> dict[str, np.ndarray]:
    base = np.arange(1, rows + 1, dtype=np.float64)
    return {
        name: base * (index + 1) / 10.0
        for index, name in enumerate(LOG_COLUMNS)
    }


def test_names_lengths_order_and_repeatability():
    assert len(LOG_COLUMNS) == 39
    assert len(PRODUCT_BASE) == 8
    assert len(PRODUCT_PAIRS) == 28
    assert SET_NAMES == ("HMOSXL", "HMOSXLP")
    assert PRODUCT_PAIRS[0] == ("implied_move", "mean_prior_abs_move")
    assert PRODUCT_PAIRS[-1] == ("iv30", "season_heat_mean_abs_20")

    names = derived_names("HMOSXLP")
    expected = tuple(f"log1p_{name}" for name in LOG_COLUMNS) + tuple(
        f"{left}_x_{right}" for left, right in PRODUCT_PAIRS
    )
    assert names == expected
    assert len(names) == len(set(names)) == 67
    assert names[0] == "log1p_mean_prior_abs_move"
    assert names[38] == "log1p_own_absret1"
    assert names[39] == "implied_move_x_mean_prior_abs_move"
    assert names[-1] == "iv30_x_season_heat_mean_abs_20"
    assert len(derived_names("HMOSXL")) == 39

    frame = _sources()
    first = derive_columns(frame, "HMOSXLP")
    second = derive_columns(frame, "HMOSXLP")
    assert list(first) == list(names)
    assert list(second) == list(names)
    for name in names:
        assert first[name].dtype == np.float64
        np.testing.assert_array_equal(first[name], second[name])


def test_logs_use_log1p_and_products_use_raw_values():
    frame = _sources()
    output = derive_columns(frame, "HMOSXLP")

    for name in LOG_COLUMNS:
        np.testing.assert_allclose(output[f"log1p_{name}"], np.log1p(frame[name]))
    for left, right in PRODUCT_PAIRS:
        np.testing.assert_allclose(
            output[f"{left}_x_{right}"], frame[left] * frame[right]
        )
    product = output["implied_move_x_mean_prior_abs_move"]
    np.testing.assert_allclose(
        product, frame["implied_move"] * frame["mean_prior_abs_move"]
    )
    assert not np.allclose(
        product,
        np.log1p(frame["implied_move"]) * np.log1p(frame["mean_prior_abs_move"]),
    )


def test_nan_stays_nan_and_is_not_zero_filled():
    frame = _sources()
    frame["mean_prior_abs_move"][0] = np.nan
    output = derive_columns(frame, "HMOSXLP")

    log = output["log1p_mean_prior_abs_move"]
    assert np.isnan(log[0])
    assert np.all(np.isfinite(log[1:]))
    assert not np.any(log[1:] == 0)

    for left, right in PRODUCT_PAIRS:
        if "mean_prior_abs_move" not in (left, right):
            continue
        product = output[f"{left}_x_{right}"]
        assert np.isnan(product[0])
        assert np.all(np.isfinite(product[1:]))


def test_negative_source_fails_contract():
    frame = _sources()
    frame["iv10"][1] = -0.1

    with pytest.raises(DataError) as caught:
        derive_columns(frame, "HMOSXL")

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert "iv10" in str(caught.value)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_source_fails_contract(value):
    frame = _sources()
    frame["iv30"][2] = value

    with pytest.raises(DataError) as caught:
        derive_columns(frame, "HMOSXLP")

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert "iv30" in str(caught.value)


def test_product_overflow_fails_only_the_product_set():
    frame = _sources()
    frame["implied_move"][0] = 1e200
    frame["mean_prior_abs_move"][0] = 1e200

    with pytest.raises(DataError) as caught:
        derive_columns(frame, "HMOSXLP")

    assert caught.value.code == "EXPERIMENT_VARIANT_FAILED"

    output = derive_columns(frame, "HMOSXL")
    assert len(output) == 39
    assert np.all(np.isfinite(output["log1p_implied_move"]))


@pytest.mark.parametrize("kind", ["missing", "length", "dtype"])
def test_invalid_sources_fail_contract(kind):
    frame = _sources()
    if kind == "missing":
        frame.pop("iv10")
    elif kind == "length":
        frame["iv10"] = frame["iv10"][:2]
    else:
        frame["iv10"] = frame["iv10"].astype(np.int64)

    with pytest.raises(DataError) as caught:
        derive_columns(frame, "HMOSXL")

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert "iv10" in str(caught.value)


@pytest.mark.parametrize(
    "bad_source",
    [[1.0, 2.0, 3.0], np.ones((3, 1), dtype=np.float64)],
    ids=["list", "2d"],
)
def test_non_1d_array_source_fails_contract(bad_source):
    frame = _sources()
    frame["iv10"] = bad_source

    with pytest.raises(DataError) as caught:
        derive_columns(frame, "HMOSXL")

    assert caught.value.code == "CONTRACT_MISMATCH"
    assert "iv10" in str(caught.value)


@pytest.mark.parametrize("set_name", ["UNKNOWN", ""])
def test_unknown_set_name_fails_contract_mismatch(set_name):
    with pytest.raises(DataError) as caught:
        derived_names(set_name)
    assert caught.value.code == "CONTRACT_MISMATCH"

    with pytest.raises(DataError) as caught:
        derive_columns(_sources(), set_name)
    assert caught.value.code == "CONTRACT_MISMATCH"


def test_input_arrays_are_not_modified():
    frame = _sources()
    references = {name: frame[name] for name in LOG_COLUMNS}
    before = {name: frame[name].copy() for name in LOG_COLUMNS}

    derive_columns(frame, "HMOSXLP")

    for name in LOG_COLUMNS:
        assert frame[name] is references[name]
        np.testing.assert_array_equal(frame[name], before[name])


def test_season_heat_ratio_is_a_nonnegative_log_source():
    assert "season_heat_ratio_20_252" in LOG_COLUMNS
    frame = _sources()
    frame["season_heat_ratio_20_252"] = np.array([0.0, 1.0, 2.0])

    output = derive_columns(frame, "HMOSXL")

    np.testing.assert_allclose(
        output["log1p_season_heat_ratio_20_252"], np.log1p([0.0, 1.0, 2.0])
    )
