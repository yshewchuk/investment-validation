from copy import deepcopy

import pytest

from engine.v2.scoring.native_analog import (
    AnalogRecipe,
    AnalogRefusal,
    evaluate_analogs,
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
