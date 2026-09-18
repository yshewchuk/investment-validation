from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from engine.analogs import AnalogMatcher
from engine.v2.scoring.native_analog import (
    AnalogRecipe,
    AnalogRefusal,
    LegacyBucketRecipe,
    bucket_population_hash,
    evaluate_analogs,
    legacy_bucket_bootstrap_seed,
    source_population_hash,
)


def _population():
    return [
        {"row_id": "b", "features": {"move": -1.0}, "realized_pnl": -1.0},
        {"row_id": "a", "features": {"move": 1.0}, "realized_pnl": 3.0},
        {"row_id": "c", "features": {"move": 1.0}, "realized_pnl": 100.0},
    ]


def _recipe(rows):
    return AnalogRecipe(
        feature_names=("move",),
        neighbors=2,
        population_hash=source_population_hash(rows),
    )


def test_nearest_neighbor_order_is_deterministic_under_input_shuffle():
    rows = _population()
    recipe = _recipe(rows)

    forward = evaluate_analogs(
        source_rows=rows,
        query_features={"move": 0.0},
        recipe=recipe,
    )
    reversed_result = evaluate_analogs(
        source_rows=list(reversed(rows)),
        query_features={"move": 0.0},
        recipe=recipe,
    )

    assert forward == reversed_result
    assert forward.exp_pnl_analog == pytest.approx(1.0)
    assert forward.win_analog == pytest.approx(0.5)
    assert forward.ci_low == pytest.approx(-2.92)
    assert forward.ci_high == pytest.approx(4.92)
    assert forward.n_analogs == 2


def test_nearest_neighbor_preserves_non_answer_feature_names():
    rows = [
        {"row_id": "a", "features": {"mean": 1.0}, "realized_pnl": 2.0},
        {"row_id": "b", "features": {"mean": 2.0}, "realized_pnl": 4.0},
    ]
    result = evaluate_analogs(
        source_rows=rows,
        query_features={"mean": 1.0},
        recipe=AnalogRecipe(
            feature_names=("mean",),
            neighbors=1,
            population_hash=source_population_hash(rows),
        ),
    )

    assert result.exp_pnl_analog == pytest.approx(2.0)


def test_missing_population_is_an_explicit_refusal():
    with pytest.raises(AnalogRefusal) as error:
        evaluate_analogs(
            source_rows=[],
            query_features={"move": 0.0},
            recipe={
                "feature_names": ["move"],
                "neighbors": 2,
                "population_hash": "sha256:" + "0" * 64,
            },
        )

    assert error.value.code == "MISSING_ANALOG_POPULATION"


def test_population_corruption_is_detected_against_recipe_hash():
    rows = _population()
    recipe = _recipe(rows)
    corrupted = deepcopy(rows)
    corrupted[0]["realized_pnl"] = 999.0

    with pytest.raises(AnalogRefusal) as error:
        evaluate_analogs(
            source_rows=corrupted,
            query_features={"move": 0.0},
            recipe=recipe,
        )

    assert error.value.code == "ANALOG_POPULATION_CORRUPT"


@pytest.mark.parametrize("location", ["source", "recipe"])
def test_precomputed_answer_fields_are_rejected(location):
    rows = _population()
    recipe = {
        "feature_names": ["move"],
        "neighbors": 2,
        "population_hash": source_population_hash(rows),
    }
    if location == "source":
        rows[0]["exp_pnl_analog"] = 7.0
    else:
        recipe["win_analog"] = 0.9

    with pytest.raises(AnalogRefusal) as error:
        evaluate_analogs(
            source_rows=rows,
            query_features={"move": 0.0},
            recipe=recipe,
        )

    assert error.value.code == "PRECOMPUTED_ANALOG_ANSWER"


BUCKET_DIMENSIONS = (
    "mcap_bucket",
    "moneyness_band",
    "dte_band",
    "implied_tercile",
)
WIDENING_ORDER = (
    "moneyness_band",
    "dte_band",
    "implied_tercile",
)


def _bucket_frame():
    common = {
        "strategy": "STR-THRU",
        "fill_alpha": 0.5,
        "mcap_bucket": "1-10B",
        "dte_band": "4-10",
        "implied_tercile": "mid",
    }
    rows = [
        common | {
            "event_id": "exact-a",
            "moneyness_band": "ATM",
            "ret": 0.10,
            "exit_date": "2020-01-02",
        },
        common | {
            "event_id": "exact-b",
            "moneyness_band": "ATM",
            "ret": 0.20,
            "exit_date": "2020-01-03",
        },
        common | {
            "event_id": "wide-a",
            "moneyness_band": "2-5%",
            "ret": -0.10,
            "exit_date": "2020-01-04",
        },
        common | {
            "event_id": "wide-b",
            "moneyness_band": "2-5%",
            "ret": 0.30,
            "exit_date": "2020-01-05",
        },
        common | {
            "event_id": "wide-missing",
            "moneyness_band": "2-5%",
            "ret": np.nan,
            "exit_date": "2020-01-06",
        },
        common | {
            "event_id": "future",
            "moneyness_band": "ATM",
            "ret": 9.99,
            "exit_date": "2025-01-02",
        },
    ]
    frame = pd.DataFrame(rows)
    frame["exit_date"] = pd.to_datetime(frame["exit_date"])
    return frame


def _bucket_query():
    return {
        "mcap_bucket": "1-10B",
        "moneyness_band": "ATM",
        "dte_band": "4-10",
        "implied_tercile": "mid",
        "implied_ratio": 1.0,
    }


def _causal_source_rows(frame):
    causal = frame[frame["exit_date"] < pd.Timestamp("2021-01-01")]
    return [
        {
            "row_id": row.event_id,
            **{dimension: getattr(row, dimension) for dimension in BUCKET_DIMENSIONS},
            "realized_return": (
                None if pd.isna(row.ret) else float(row.ret)
            ),
        }
        for row in causal.itertuples(index=False)
    ]


def _bucket_recipe(source_rows, *, min_analogs, bootstrap_draws):
    buckets = _bucket_query()
    return LegacyBucketRecipe(
        bucket_dimensions=BUCKET_DIMENSIONS,
        widening_order=WIDENING_ORDER,
        min_analogs=min_analogs,
        alpha=0.5,
        bootstrap_draws=bootstrap_draws,
        bootstrap_seed=legacy_bucket_bootstrap_seed(
            snapshot="snapshot-123",
            strategy="STR-THRU",
            alpha=0.5,
            buckets=buckets,
            request_key="request-7",
        ),
        ci_quantiles=(0.05, 0.95),
        population_hash=bucket_population_hash(source_rows, BUCKET_DIMENSIONS),
    )


def _legacy_match(frame, *, min_analogs, bootstrap_draws):
    captured = []
    result = AnalogMatcher(frame, snapshot="snapshot-123").match(
        "STR-THRU",
        _bucket_query(),
        alpha=0.5,
        as_of="2021-01-01",
        min_analogs=min_analogs,
        bootstrap=bootstrap_draws,
        request_key="request-7",
        evidence_hook=captured.append,
    )
    return result, captured[0]


def _assert_legacy_summary(native, legacy):
    assert native.exp_pnl_analog == pytest.approx(legacy.mean)
    assert native.win_analog == pytest.approx(legacy.win_rate)
    assert native.ci_low == pytest.approx(legacy.ci_low)
    assert native.ci_high == pytest.approx(legacy.ci_high)
    assert native.n_analogs == legacy.n
    assert native.median == pytest.approx(legacy.median)
    assert native.p10 == pytest.approx(legacy.p10)
    assert native.p90 == pytest.approx(legacy.p90)
    assert native.widened == legacy.widened
    assert native.dropped == legacy.dropped
    assert native.thin == legacy.thin


def test_bucket_recipe_reproduces_legacy_exact_match_and_causal_population():
    frame = _bucket_frame()
    source_rows = _causal_source_rows(frame)
    legacy, evidence = _legacy_match(frame, min_analogs=2, bootstrap_draws=128)

    native = evaluate_analogs(
        source_rows=source_rows,
        query_features={key: _bucket_query()[key] for key in BUCKET_DIMENSIONS},
        recipe=_bucket_recipe(source_rows, min_analogs=2, bootstrap_draws=128),
    )

    _assert_legacy_summary(native, legacy)
    legacy_selected = {
        row["values"]["event_id"] for row in evidence["selected"]["rows"]
    }
    assert set(native.selected_row_ids) == legacy_selected
    assert set(native.population_row_ids) == {
        row["event_id"] for row in frame.to_dict("records")
        if row["exit_date"] < pd.Timestamp("2021-01-01")
    }


def test_bucket_recipe_reproduces_legacy_fixed_order_widening():
    frame = _bucket_frame()
    source_rows = _causal_source_rows(frame)
    legacy, evidence = _legacy_match(frame, min_analogs=5, bootstrap_draws=128)

    serialized_recipe = json.loads(json.dumps(vars(
        _bucket_recipe(source_rows, min_analogs=5, bootstrap_draws=128)
    )))
    native = evaluate_analogs(
        source_rows=list(reversed(source_rows)),
        query_features={key: _bucket_query()[key] for key in BUCKET_DIMENSIONS},
        recipe=serialized_recipe,
    )

    _assert_legacy_summary(native, legacy)
    legacy_selected = {
        row["values"]["event_id"] for row in evidence["selected"]["rows"]
    }
    assert set(native.selected_row_ids) == legacy_selected
    assert native.contributing_row_ids == tuple(
        row_id for row_id in native.selected_row_ids if row_id != "wide-missing"
    )


def test_bucket_population_hash_is_deterministic_under_input_shuffle():
    rows = _causal_source_rows(_bucket_frame())

    assert bucket_population_hash(rows, BUCKET_DIMENSIONS) == (
        bucket_population_hash(list(reversed(rows)), BUCKET_DIMENSIONS)
    )


def test_bucket_recipe_matches_legacy_empty_causal_population():
    frame = _bucket_frame()
    frame["exit_date"] = pd.Timestamp("2025-01-02")
    source_rows = []
    legacy, evidence = _legacy_match(frame, min_analogs=2, bootstrap_draws=128)

    native = evaluate_analogs(
        source_rows=source_rows,
        query_features={key: _bucket_query()[key] for key in BUCKET_DIMENSIONS},
        recipe=_bucket_recipe(source_rows, min_analogs=2, bootstrap_draws=128),
    )

    _assert_legacy_summary(native, legacy)
    assert native.population_row_ids == ()
    assert native.selected_row_ids == ()
    assert evidence["causal"]["row_ids"] == []


@pytest.mark.parametrize("location", ["source", "recipe", "query"])
def test_bucket_recipe_rejects_calculated_answer_fields(location):
    rows = _causal_source_rows(_bucket_frame())
    recipe = vars(_bucket_recipe(rows, min_analogs=2, bootstrap_draws=0)).copy()
    query = {key: _bucket_query()[key] for key in BUCKET_DIMENSIONS}
    if location == "source":
        rows[0]["analog_mean"] = 0.99
    elif location == "recipe":
        recipe["win_analog"] = 0.99
    else:
        query["n_analogs"] = 999

    with pytest.raises(AnalogRefusal) as error:
        evaluate_analogs(
            source_rows=rows,
            query_features=query,
            recipe=recipe,
        )

    assert error.value.code == "PRECOMPUTED_ANALOG_ANSWER"
