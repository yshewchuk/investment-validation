"""Derived rows feed the fit path: detection, derivation, wiring and failure codes."""
import copy
from pathlib import Path

import numpy as np
import pytest

from engine.v2.data.errors import DataError
from engine.v2.foundation import ArtifactStore
from engine.v2.research.derived_features import LOG_COLUMNS, PRODUCT_PAIRS, derived_names
from engine.v2.research.horizon_fitted import build_fit_inputs
from experiments.derived_rows import with_derived
from experiments.magnitude_search_evidence import fit_evidence
from experiments.magnitude_search_inputs import load_fit_inputs, publish_inputs
from tests.v2.research.test_magnitude_search_inputs import _fit_cohort

HORIZONS = [1, 3]
STORED = ("x", *LOG_COLUMNS)


def _cohort():
    cohort, events = _fit_cohort(3)
    for index, row in enumerate(cohort["features"]):
        event = events.index(row["event_id"])
        for name in LOG_COLUMNS:
            row[name] = 0.25 + event + 0.01 * (index + 1)
    return cohort, events


def _columns(set_name):
    return (*STORED, *derived_names(set_name))


def _fit(store, cohort, events, columns):
    manifest, inputs_id = publish_inputs(store, cohort)
    return load_fit_inputs(store, manifest, inputs_id, [events[2]], HORIZONS, columns,
                           numeric_fit_inputs=True)


def _raw_index(cohort):
    return {(row["event_id"], row["horizon_sessions"]): row for row in cohort["features"]}


def _inputs(cohort):
    return {"features": cohort["features"], "training": cohort["training"]}


def _raises(call, code):
    with pytest.raises(DataError) as caught:
        call()
    assert caught.value.code == code


def test_stored_only_columns_stay_the_same_object_and_fit_identically(tmp_path):
    cohort, events = _cohort()
    rows = cohort["features"]
    assert with_derived(rows, STORED) is rows
    loaded = _fit(ArtifactStore(tmp_path), cohort, events, STORED)
    expected = build_fit_inputs(_raw_index(cohort), [row["event_id"] for row in cohort["training"]],
                                [events[2]], HORIZONS, list(STORED),
                                [row["absolute_move_pct"] for row in cohort["training"]],
                                target_column="absolute_move_pct")
    for horizon in HORIZONS:
        actual = loaded.fit_inputs[horizon]
        assert np.array_equal(actual.train_features, expected[horizon].train_features)
        assert actual.train_features.tolist() == expected[horizon].train_features.tolist()
        assert np.array_equal(actual.prediction_features, expected[horizon].prediction_features)
        assert actual.prediction_features.tolist() == expected[horizon].prediction_features.tolist()


@pytest.mark.parametrize("set_name, count", [("HMOSXL", 39), ("HMOSXLP", 67)])
def test_derived_matrices_keep_stored_columns_and_add_the_set(tmp_path, set_name, count):
    cohort, events = _cohort()
    columns = _columns(set_name)
    loaded = _fit(ArtifactStore(tmp_path), cohort, events, columns)
    raw = _raw_index(cohort)
    derived = {}
    for row in with_derived(cohort["features"], columns):
        key = (row["event_id"], row["horizon_sessions"])
        derived[key] = {name: row[name]
                        for name in ("event_id", "context_id", "horizon_sessions", *columns)}
    assert len(derived_names(set_name)) == count
    for horizon in HORIZONS:
        assert loaded.fit_inputs[horizon].columns == list(columns)
    got = {(row["event_id"], row["horizon_sessions"]): row for row in loaded["features"]}
    assert set(got) == {(event, horizon) for event in events for horizon in HORIZONS}
    for key, row in got.items():
        assert row == derived[key]
    key = (events[0], 1)
    sample, source = got[key], raw[key]
    for name in STORED:
        assert sample[name] == source[name]
    for name in LOG_COLUMNS:
        assert sample[f"log1p_{name}"] == pytest.approx(np.log1p(source[name]))
    if set_name == "HMOSXLP":
        for left, right in PRODUCT_PAIRS:
            assert sample[f"{left}_x_{right}"] == pytest.approx(source[left] * source[right])


def test_null_sources_stay_null_and_do_not_refuse(tmp_path):
    cohort, events = _cohort()
    cohort["features"][0]["mean_prior_abs_move"] = None
    columns = _columns("HMOSXLP")
    derived = with_derived(cohort["features"], columns)[0]
    assert derived["log1p_mean_prior_abs_move"] is None
    for left, right in PRODUCT_PAIRS:
        if "mean_prior_abs_move" in (left, right):
            assert derived[f"{left}_x_{right}"] is None
    assert derived["log1p_iv10"] is not None
    loaded = _fit(ArtifactStore(tmp_path), cohort, events, columns)
    assert loaded["features"][0]["log1p_mean_prior_abs_move"] is None


def _corrupt(cohort, kind):
    row = cohort["features"][0]
    if kind == "negative":
        row["iv10"] = -0.1
    elif kind == "infinite":
        row["iv30"] = np.inf
    else:
        row["implied_move"] = 1e200
        row["mean_prior_abs_move"] = 1e200


@pytest.mark.parametrize("kind, code", [
    ("negative", "CONTRACT_MISMATCH"),
    ("infinite", "CONTRACT_MISMATCH"),
    ("overflow", "EXPERIMENT_VARIANT_FAILED"),
])
def test_failure_rows_refuse_through_every_entry(tmp_path, kind, code):
    cohort, events = _cohort()
    _corrupt(cohort, kind)
    columns = _columns("HMOSXLP" if kind == "overflow" else "HMOSXL")
    _raises(lambda: with_derived(cohort["features"], columns), code)
    _raises(lambda: _fit(ArtifactStore(tmp_path), cohort, events, columns), code)
    _raises(lambda: fit_evidence(_inputs(cohort), {"evaluation_event_ids": [events[2]]},
                                 {"horizons": HORIZONS}, {"features": list(columns)}), code)


def test_missing_derived_source_refuses_through_every_entry(tmp_path):
    cohort, events = _cohort()
    columns = ("x", *(name for name in LOG_COLUMNS if name != "iv10"),
               *derived_names("HMOSXL"))
    _raises(lambda: with_derived(cohort["features"], columns), "CONTRACT_MISMATCH")
    _raises(lambda: _fit(ArtifactStore(tmp_path), cohort, events, columns), "CONTRACT_MISMATCH")
    _raises(lambda: fit_evidence(_inputs(cohort), {"evaluation_event_ids": [events[2]]},
                                 {"horizons": HORIZONS}, {"features": list(columns)}),
            "CONTRACT_MISMATCH")


def test_hmosxl_evidence_matches_the_parent_build_and_leaves_rows_alone():
    cohort, events = _cohort()
    columns = _columns("HMOSXL")
    before = copy.deepcopy(cohort["features"])
    evidence = fit_evidence(_inputs(cohort), {"evaluation_event_ids": [events[2]]},
                            {"horizons": HORIZONS}, {"features": list(columns)})
    assert cohort["features"] == before
    expected = build_fit_inputs(
        {(row["event_id"], row["horizon_sessions"]): row
         for row in with_derived(cohort["features"], columns)},
        [row["event_id"] for row in cohort["training"]], [events[2]], HORIZONS,
        list(columns), [row["absolute_move_pct"] for row in cohort["training"]],
        target_column="absolute_move_pct")
    for item, horizon in zip(evidence, HORIZONS):
        assert item["training_matrix_id"] == expected.evidence[horizon]["training_matrix_id"]
        assert item["prediction_matrix_id"] == expected.evidence[horizon]["prediction_matrix_id"]


def test_one_progress_line_reports_the_derived_set(capsys):
    cohort, _ = _cohort()
    with_derived(cohort["features"], _columns("HMOSXL"))
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("[derive]")]
    assert len(lines) == 1
    for part in ("set=HMOSXL", "rows=", "derived=39", "nulls=", "rss="):
        assert part in lines[0]


def test_worker_maps_the_variant_failure_to_failed():
    root = Path(__file__).resolve().parents[3]
    source = (root / "experiments/magnitude_search_worker.py").read_text()
    assert "from engine.v2.data.errors import DataError" in source
    assert "except (OpsError, DataError) as error:" in source
    assert 'if error.code == "EXPERIMENT_VARIANT_FAILED":' in source
    assert 'emit({"event": "failed"})' in source


def test_v12_panel_columns_equal_the_v11_stored_columns():
    from engine.v2.research.magnitude_panel_qualified_table import recipe
    from experiments.size_search_binding import _panel_columns, _panel_subset
    v12, v11 = "magnitude_search_request.v12", "magnitude_search_request.v11"
    assert _panel_subset(v12) == "HMOSX"
    assert (_panel_columns(v12, recipe(_panel_subset(v12))["model_columns"])
            == _panel_columns(v11, recipe(_panel_subset(v11))["model_columns"]))
    assert not set(derived_names("HMOSXLP")) & set(_panel_columns(v12, recipe("HMOSX")["model_columns"]))
