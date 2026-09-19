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


# ---------------------------------------------------------------------------
# mutation-pilot triage
#
# 1. Legacy parity over a population and query set wide enough to reach every
#    branch: zero returns, a missing return, a thin match, widening to the
#    end, a partly and a fully unavailable query, and a query with no match.
# 2. The nearest-neighbour arithmetic, pinned by hand-computed values.
# 3. Every reachable refusal, by its stable code. stages.py turns an
#    AnalogRefusal into the record flag "CODE: detail", so the code is output.
# ---------------------------------------------------------------------------

_LABELS = {
    "mcap_bucket": ("<1B", "1-10B", ">=10B"),
    "moneyness_band": ("ATM", "2-5%", ">5%"),
    "dte_band": ("1-3", "4-10", "11-25"),
    "implied_tercile": ("low", "mid", "high"),
}


def _parity_frame():
    rng = np.random.default_rng(11)
    rows = []
    for index in range(90):
        ret = float(rng.choice([0.0, -0.05, 0.02, 0.08, -0.12, 0.15, np.nan]))
        rows.append({
            "strategy": "STR-THRU", "fill_alpha": 0.5, "event_id": f"e{index:03d}",
            **{dim: str(rng.choice(labels[:2])) for dim, labels in _LABELS.items()},
            "ret": ret, "exit_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index),
        })
    return pd.DataFrame(rows)


_PARITY_QUERIES = [
    ({"mcap_bucket": "<1B", "moneyness_band": "ATM", "dte_band": "1-3",
      "implied_tercile": "low"}, 3),
    ({"mcap_bucket": "<1B", "moneyness_band": "ATM", "dte_band": "1-3",
      "implied_tercile": "low"}, 12),
    ({"mcap_bucket": "1-10B", "moneyness_band": "2-5%", "dte_band": "4-10",
      "implied_tercile": "mid"}, 1),
    ({"mcap_bucket": "<1B", "moneyness_band": "ATM", "dte_band": "1-3",
      "implied_tercile": "low"}, 500),
    ({"mcap_bucket": "1-10B", "moneyness_band": "ATM", "dte_band": None,
      "implied_tercile": "mid"}, 4),
    ({"mcap_bucket": None, "moneyness_band": None, "dte_band": None,
      "implied_tercile": None}, 4),
    ({"mcap_bucket": ">=10B", "moneyness_band": "ATM", "dte_band": "1-3",
      "implied_tercile": "low"}, 4),
]


@pytest.mark.parametrize("query, min_analogs", _PARITY_QUERIES)
def test_bucket_recipe_matches_legacy_across_widening_and_unavailable_queries(
    query, min_analogs,
):
    frame = _parity_frame()
    legacy = AnalogMatcher(frame, snapshot="snap-9").match(
        "STR-THRU", dict(query), alpha=0.5, min_analogs=min_analogs, bootstrap=64,
        request_key="req-3",
    )
    source_rows = [
        {"row_id": row.event_id,
         **{dim: getattr(row, dim) for dim in BUCKET_DIMENSIONS},
         "realized_return": None if pd.isna(row.ret) else float(row.ret)}
        for row in frame.itertuples(index=False)
    ]
    recipe = LegacyBucketRecipe(
        bucket_dimensions=BUCKET_DIMENSIONS, widening_order=WIDENING_ORDER,
        min_analogs=min_analogs, alpha=0.5, bootstrap_draws=64,
        bootstrap_seed=legacy_bucket_bootstrap_seed(
            snapshot="snap-9", strategy="STR-THRU", alpha=0.5, buckets=query,
            request_key="req-3"),
        ci_quantiles=(0.05, 0.95),
        population_hash=bucket_population_hash(source_rows, BUCKET_DIMENSIONS),
    )
    native = evaluate_analogs(source_rows=source_rows, query_features=query, recipe=recipe)

    _assert_legacy_summary(native, legacy)
    assert native.unavailable == legacy.unavailable
    if legacy.n:
        assert native.win_analog == legacy.win_rate  # exact: 0.0 is not a win
    matched = frame
    for dim in BUCKET_DIMENSIONS:
        if dim not in native.dropped:
            matched = matched[matched[dim] == query[dim]]
    if len(native.unavailable) == len(BUCKET_DIMENSIONS):
        matched = matched.iloc[0:0]
    assert native.selected_row_ids == tuple(sorted(matched["event_id"]))
    assert native.contributing_row_ids == tuple(
        sorted(matched.loc[matched["ret"].notna(), "event_id"]))
    assert native.population_row_ids == tuple(sorted(frame["event_id"]))


def test_parity_population_holds_zero_returns_and_thin_matches():
    """Guards the fixture above: it must keep exercising the branches."""
    frame = _parity_frame()
    assert (frame["ret"] == 0.0).any() and frame["ret"].isna().any()


def _nearest(rows, query, neighbors, names=("x", "y")):
    return evaluate_analogs(
        source_rows=rows, query_features=query,
        recipe=AnalogRecipe(feature_names=names, neighbors=neighbors,
                            population_hash=source_population_hash(rows)),
    )


def test_nearest_neighbours_rank_by_squared_euclidean_distance():
    rows = [
        {"row_id": "far", "features": {"x": -3.0, "y": 0.0}, "realized_pnl": 10.0},
        {"row_id": "mid", "features": {"x": 2.0, "y": 0.0}, "realized_pnl": 20.0},
        {"row_id": "near", "features": {"x": 1.0, "y": 0.5}, "realized_pnl": -5.0},
    ]
    # distances^2 from (0.5, 0.5): far 12.25, mid 2.5, near 0.25
    result = _nearest(rows, {"x": 0.5, "y": 0.5}, 2)
    assert result.exp_pnl_analog == 7.5
    assert result.n_analogs == 2


def test_nearest_neighbour_ties_break_by_row_id_not_outcome():
    rows = [
        {"row_id": "b", "features": {"x": 1.0, "y": 0.0}, "realized_pnl": -1.0},
        {"row_id": "a", "features": {"x": -1.0, "y": 0.0}, "realized_pnl": 5.0},
        {"row_id": "0", "features": {"x": 9.0, "y": 0.0}, "realized_pnl": 7.0},
    ]
    assert _nearest(rows, {"x": 0.0, "y": 0.0}, 1).exp_pnl_analog == 5.0


def test_nearest_neighbour_statistics_by_hand():
    rows = [
        {"row_id": "a", "features": {"x": 0.0, "y": 0.0}, "realized_pnl": 0.0},
        {"row_id": "b", "features": {"x": 0.1, "y": 0.0}, "realized_pnl": 0.5},
        {"row_id": "c", "features": {"x": 0.2, "y": 0.0}, "realized_pnl": 2.5},
    ]
    result = _nearest(rows, {"x": 0.0, "y": 0.0}, 3)
    mean = 1.0
    variance = ((0.0 - 1) ** 2 + (0.5 - 1) ** 2 + (2.5 - 1) ** 2) / 2  # 1.75
    margin = 1.96 * (variance / 3) ** 0.5
    assert result.exp_pnl_analog == pytest.approx(mean)
    assert result.win_analog == pytest.approx(2 / 3)  # 0.0 is not a win
    assert result.ci_low == pytest.approx(mean - margin)
    assert result.ci_high == pytest.approx(mean + margin)
    single = _nearest(rows, {"x": 0.0, "y": 0.0}, 1)
    assert (single.exp_pnl_analog, single.ci_low, single.ci_high) == (0.0, 0.0, 0.0)


# -- refusals by code -------------------------------------------------------

_HASH = "sha256:" + "0" * 64


def _nearest_case(rows=None, query=None, recipe=None):
    rows = _population() if rows is None else rows
    base = {"feature_names": ["move"], "neighbors": 2,
            "population_hash": source_population_hash(_population())}
    return rows, ({"move": 0.0} if query is None else query), (
        base if recipe is None else recipe)


def _with(mapping, **changes):
    out = dict(mapping)
    for key, value in changes.items():
        if value is _DROP:
            out.pop(key, None)
        else:
            out[key] = value
    return out


_DROP = object()
_NR = {"feature_names": ["move"], "neighbors": 2, "population_hash": _HASH}
_ROW = {"row_id": "z", "features": {"move": 0.0}, "realized_pnl": 1.0}

_NEAREST_REFUSALS = {
    # recipe
    "recipe not a mapping": (None, None, ["move"], "INVALID_ANALOG_RECIPE"),
    "recipe unsupported key": (None, None, _with(_NR, extra=1), "INVALID_ANALOG_RECIPE"),
    "recipe missing key": (None, None, _with(_NR, neighbors=_DROP), "INVALID_ANALOG_RECIPE"),
    "recipe bucket-only answer": (None, None, _with(_NR, analog_mean=1.0),
                                  "PRECOMPUTED_ANALOG_ANSWER"),
    "recipe nested answer": (None, None, _with(_NR, feature_names=[{"win_analog": 1}]),
                             "PRECOMPUTED_ANALOG_ANSWER"),
    "recipe schema": (None, None, _with(_NR, schema_version="native_analog_recipe.v9"),
                      "INVALID_ANALOG_RECIPE"),
    "object schema": (None, None, AnalogRecipe(feature_names=("move",), neighbors=2,
                                               population_hash=_HASH, schema_version="v9"),
                      "INVALID_ANALOG_RECIPE"),
    "no feature names": (None, None, _with(_NR, feature_names=[]), "INVALID_ANALOG_RECIPE"),
    "blank feature name": (None, None, _with(_NR, feature_names=["move", " "]),
                           "INVALID_ANALOG_RECIPE"),
    "duplicate feature": (None, None, _with(_NR, feature_names=["move", "move"]),
                          "INVALID_ANALOG_RECIPE"),
    "neighbors zero": (None, None, _with(_NR, neighbors=0), "INVALID_ANALOG_RECIPE"),
    "neighbors bool": (None, None, _with(_NR, neighbors=True), "INVALID_ANALOG_RECIPE"),
    "neighbors float": (None, None, _with(_NR, neighbors=2.0), "INVALID_ANALOG_RECIPE"),
    "hash not content hash": (None, None, _with(_NR, population_hash="md5:1"),
                              "INVALID_ANALOG_RECIPE"),
    # rows
    "row not a mapping": ([["z"]], None, _NR, "INVALID_ANALOG_SOURCE"),
    "row unsupported field": ([_with(_ROW, extra=1)], None, _NR, "INVALID_ANALOG_SOURCE"),
    "row missing field": ([_with(_ROW, realized_pnl=_DROP)], None, _NR,
                          "INVALID_ANALOG_SOURCE"),
    "row empty id": ([_with(_ROW, row_id="  ")], None, _NR, "INVALID_ANALOG_SOURCE"),
    "row duplicate id": ([_ROW, dict(_ROW)], None, _NR, "INVALID_ANALOG_SOURCE"),
    "row features not mapping": ([_with(_ROW, features=[1.0])], None, _NR,
                                 "INVALID_ANALOG_SOURCE"),
    "row nested answer": ([_with(_ROW, features={"move": 0.0, "x": {"n_analogs": 3}})],
                          None, _NR, "PRECOMPUTED_ANALOG_ANSWER"),
    "row feature inf": ([_with(_ROW, features={"move": float("inf")})], None, _NR,
                        "INVALID_ANALOG_INPUT"),
    "row feature bool": ([_with(_ROW, features={"move": True})], None, _NR,
                         "INVALID_ANALOG_INPUT"),
    "row feature text": ([_with(_ROW, features={"move": "x"})], None, _NR,
                         "INVALID_ANALOG_INPUT"),
    "row pnl nan": ([_with(_ROW, realized_pnl=float("nan"))], None, _NR,
                    "INVALID_ANALOG_INPUT"),
    # query
    "query not a mapping": (None, [0.0], "valid", "INVALID_ANALOG_INPUT"),
    "query answer": (None, {"move": 0.0, "win_analog": 1.0}, "valid",
                     "PRECOMPUTED_ANALOG_ANSWER"),
    "query missing feature": (None, {"other": 0.0}, "valid", "MISSING_ANALOG_FEATURE"),
    "query feature nan": (None, {"move": float("nan")}, "valid", "INVALID_ANALOG_INPUT"),
    "row missing feature": ([_with(_ROW, features={"other": 1.0})], None, "hashed",
                            "MISSING_ANALOG_FEATURE"),
}


@pytest.mark.parametrize("case", sorted(_NEAREST_REFUSALS))
def test_nearest_analog_refusals_carry_their_stable_code(case):
    rows, query, recipe, code = _NEAREST_REFUSALS[case]
    rows = _population() if rows is None else deepcopy(rows)
    if recipe in ("valid", "hashed"):
        recipe = {"feature_names": ["move"], "neighbors": 2,
                  "population_hash": source_population_hash(rows)}
    with pytest.raises(AnalogRefusal) as error:
        evaluate_analogs(source_rows=rows,
                         query_features={"move": 0.0} if query is None else query,
                         recipe=recipe)
    assert error.value.code == code


def _bucket_rows():
    return _causal_source_rows(_bucket_frame())


def _bucket_recipe_doc(**changes):
    rows = _bucket_rows()
    doc = vars(_bucket_recipe(rows, min_analogs=2, bootstrap_draws=16)).copy()
    doc["bucket_dimensions"] = list(doc["bucket_dimensions"])
    doc["widening_order"] = list(doc["widening_order"])
    doc["ci_quantiles"] = list(doc["ci_quantiles"])
    return _with(doc, **changes)


def _bucket_query_exact():
    return {key: _bucket_query()[key] for key in BUCKET_DIMENSIONS}


_BROW = {"row_id": "zz", "mcap_bucket": "1-10B", "moneyness_band": "ATM",
         "dte_band": "4-10", "implied_tercile": "mid", "realized_return": 0.1}

_BUCKET_REFUSALS = {
    # recipe mapping shape
    "unsupported key": (dict(extra=1), None, None, "INVALID_ANALOG_RECIPE"),
    "missing key": (dict(min_analogs=_DROP), None, None, "INVALID_ANALOG_RECIPE"),
    "quantiles text": (dict(ci_quantiles="05"), None, None, "INVALID_ANALOG_RECIPE"),
    "quantiles scalar": (dict(ci_quantiles=0.05), None, None, "INVALID_ANALOG_RECIPE"),
    "dimensions text": (dict(bucket_dimensions="mcap_bucket"), None, None,
                        "INVALID_ANALOG_RECIPE"),
    "widening empty text": (dict(widening_order=""), None, None, "INVALID_ANALOG_RECIPE"),
    "widening not a sequence": (dict(widening_order=3), None, None,
                                "INVALID_ANALOG_RECIPE"),
    # dimensions
    "no dimensions": (dict(bucket_dimensions=[], widening_order=[]), None, None,
                      "INVALID_ANALOG_RECIPE"),
    "blank dimension": (dict(bucket_dimensions=[*BUCKET_DIMENSIONS, " "]), None, None,
                        "INVALID_ANALOG_RECIPE"),
    "duplicate dimension": (dict(bucket_dimensions=[*BUCKET_DIMENSIONS, "dte_band"]),
                            None, None, "INVALID_ANALOG_RECIPE"),
    "duplicate widening": (dict(widening_order=["dte_band", "dte_band"]), None, None,
                           "INVALID_ANALOG_RECIPE"),
    "foreign widening": (dict(widening_order=["dte_band", "sector"]), None, None,
                         "INVALID_ANALOG_RECIPE"),
    # sampling
    "min_analogs zero": (dict(min_analogs=0), None, None, "INVALID_ANALOG_RECIPE"),
    "min_analogs bool": (dict(min_analogs=True), None, None, "INVALID_ANALOG_RECIPE"),
    "min_analogs float": (dict(min_analogs=2.0), None, None, "INVALID_ANALOG_RECIPE"),
    "draws negative": (dict(bootstrap_draws=-1), None, None, "INVALID_ANALOG_RECIPE"),
    "draws bool": (dict(bootstrap_draws=True), None, None, "INVALID_ANALOG_RECIPE"),
    "draws float": (dict(bootstrap_draws=16.0), None, None, "INVALID_ANALOG_RECIPE"),
    "seed negative": (dict(bootstrap_seed=-1), None, None, "INVALID_ANALOG_RECIPE"),
    "seed 2**64": (dict(bootstrap_seed=2**64), None, None, "INVALID_ANALOG_RECIPE"),
    "seed bool": (dict(bootstrap_seed=False), None, None, "INVALID_ANALOG_RECIPE"),
    "seed float": (dict(bootstrap_seed=1.0), None, None, "INVALID_ANALOG_RECIPE"),
    # statistics
    "alpha nan": (dict(alpha=float("nan")), None, None, "INVALID_ANALOG_INPUT"),
    "alpha negative": (dict(alpha=-0.1), None, None, "INVALID_ANALOG_RECIPE"),
    "alpha above one": (dict(alpha=1.5), None, None, "INVALID_ANALOG_RECIPE"),
    "three quantiles": (dict(ci_quantiles=[0.05, 0.5, 0.95]), None, None,
                        "INVALID_ANALOG_RECIPE"),
    "quantile inf": (dict(ci_quantiles=[0.05, float("inf")]), None, None,
                     "INVALID_ANALOG_INPUT"),
    "quantile low nan": (dict(ci_quantiles=[float("nan"), 0.95]), None, None,
                         "INVALID_ANALOG_INPUT"),
    "quantiles reversed": (dict(ci_quantiles=[0.95, 0.05]), None, None,
                           "INVALID_ANALOG_RECIPE"),
    "quantile below zero": (dict(ci_quantiles=[-0.1, 0.95]), None, None,
                            "INVALID_ANALOG_RECIPE"),
    "quantile above one": (dict(ci_quantiles=[0.05, 1.5]), None, None,
                           "INVALID_ANALOG_RECIPE"),
    "hash not content hash": (dict(population_hash="md5:1"), None, None,
                              "INVALID_ANALOG_RECIPE"),
    # rows
    "row not a mapping": ({}, [["zz"]], None, "INVALID_ANALOG_SOURCE"),
    "row unsupported field": ({}, [_with(_BROW, extra=1)], None, "INVALID_ANALOG_SOURCE"),
    "row missing field": ({}, [_with(_BROW, dte_band=_DROP)], None,
                          "INVALID_ANALOG_SOURCE"),
    "row empty id": ({}, [_with(_BROW, row_id="")], None, "INVALID_ANALOG_SOURCE"),
    "row duplicate id": ({}, [_BROW, dict(_BROW)], None, "INVALID_ANALOG_SOURCE"),
    "row return inf": ({}, [_with(_BROW, realized_return=float("inf"))], None,
                       "INVALID_ANALOG_INPUT"),
    "row bucket bool": ({}, [_with(_BROW, dte_band=True)], None, "INVALID_ANALOG_SOURCE"),
    "row bucket nan": ({}, [_with(_BROW, dte_band=float("nan"))], None,
                       "INVALID_ANALOG_SOURCE"),
    "row bucket list": ({}, [_with(_BROW, dte_band=["4-10"])], None,
                        "INVALID_ANALOG_SOURCE"),
    "population corrupt": ({}, [_BROW], None, "ANALOG_POPULATION_CORRUPT"),
    # query
    "query not a mapping": ({}, None, ["ATM"], "INVALID_ANALOG_INPUT"),
    "query unsupported": ({}, None, dict(_bucket_query_exact(), sector="x"),
                          "INVALID_ANALOG_INPUT"),
    "query missing": ({}, None, _with(_bucket_query_exact(), dte_band=_DROP),
                      "INVALID_ANALOG_INPUT"),
    "query bucket-only answer": ({}, None, dict(_bucket_query_exact(), thin=True),
                                 "PRECOMPUTED_ANALOG_ANSWER"),
    "query nested answer": ({}, None, dict(_bucket_query_exact(),
                                           dte_band={"analog_mean": 1.0}),
                            "PRECOMPUTED_ANALOG_ANSWER"),
    "query listed answer": ({}, None, dict(_bucket_query_exact(),
                                           dte_band=[{"widened": 1}]),
                            "PRECOMPUTED_ANALOG_ANSWER"),
    "query bucket inf": ({}, None, dict(_bucket_query_exact(), dte_band=float("inf")),
                         "INVALID_ANALOG_SOURCE"),
}


@pytest.mark.parametrize("case", sorted(_BUCKET_REFUSALS))
def test_bucket_analog_refusals_carry_their_stable_code(case):
    changes, rows, query, code = _BUCKET_REFUSALS[case]
    recipe = _bucket_recipe_doc(**changes)
    # A malformed row is refused while normalizing, before the hash check, so
    # the recipe's (fixture) population_hash only matters for "population corrupt".
    if rows is None:
        rows = _bucket_rows()
    with pytest.raises(AnalogRefusal) as error:
        evaluate_analogs(source_rows=deepcopy(rows),
                         query_features=_bucket_query_exact() if query is None else query,
                         recipe=recipe)
    assert error.value.code == code


@pytest.mark.parametrize("changes", [
    dict(min_analogs=1), dict(bootstrap_draws=0), dict(bootstrap_seed=0),
    dict(bootstrap_seed=2**64 - 1), dict(alpha=0.0), dict(alpha=1.0),
    dict(ci_quantiles=[0.0, 1.0]), dict(ci_quantiles=[0.5, 0.5]),
])
def test_bucket_recipe_boundaries_that_are_accepted(changes):
    result = evaluate_analogs(source_rows=_bucket_rows(),
                              query_features=_bucket_query_exact(),
                              recipe=_bucket_recipe_doc(**changes))
    assert result.n_analogs == 2


def test_bucket_values_may_be_finite_floats_and_ints():
    rows = [_with(_BROW, row_id=f"r{i}", implied_tercile=0.5, dte_band=7)
            for i in range(3)]
    recipe = _bucket_recipe_doc(population_hash=bucket_population_hash(rows, BUCKET_DIMENSIONS))
    query = dict(_bucket_query_exact(), implied_tercile=0.5, dte_band=7)
    assert evaluate_analogs(source_rows=rows, query_features=query,
                            recipe=recipe).n_analogs == 3
