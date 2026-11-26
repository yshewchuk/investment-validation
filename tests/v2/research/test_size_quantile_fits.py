"""EXP-190 Q3a fit stage: synthetic fixtures and fake fits; only the fit_fold test uses real sklearn."""
import numpy as np
import pytest

from engine.v2.ops.errors import OpsError
from experiments import size_quantile_fits as fits
from experiments.size_quantile_prereg import load_spec

SPEC, _ = load_spec()
LEVELS = list(SPEC["levels"])
KEYS = ("7x300", "7x1000", "15x300", "15x1000")


def _spec(max_leaf, min_leaf):
    return {**SPEC, "grid": {**SPEC["grid"], "axes": {"max_leaf_nodes": list(max_leaf),
                                                      "min_samples_leaf": list(min_leaf)}}}


def _receipt(event_ids, values):
    """values: {key: one scalar per event}; each scalar becomes a constant 7-level raw row."""
    return {"event_ids": list(event_ids), "raw": {k: [[v] * 7 for v in vals] for k, vals in values.items()}}


def _refusal(exc):
    return exc.value.problem.code, exc.value.problem.details.get("reason")


def _fit_sealed(n=60):
    rows = [{"event_id": f"f{k:02d}", "f_a": (k % 9) * 0.1 - 0.4, "f_b": None if k == 4 else k / n,
             "f_empty": None} for k in range(n)]
    return {"model_columns": ["f_a", "f_b", "f_empty"], "dictionary_hash": "sha256:" + "a" * 64, "rows": rows}


TRAINING = [(f"f{k:02d}", float(1 + k % 5)) for k in range(45)]
EVALUATION = [f"f{k:02d}" for k in range(45, 60)]


def _patch_fit(monkeypatch, fit=None):
    """Record every fit call; by default return 10 * quantile for each evaluation row."""
    calls = []

    def fake(recipe, train_x, y, eval_x, *, columns):
        calls.append({"recipe": recipe, "train_x": train_x, "eval_x": eval_x, "columns": columns})
        if fit is not None:
            return fit(recipe, len(eval_x))
        return {"predictions": [10.0 * recipe["estimator"]["params"]["quantile"]] * len(eval_x)}

    monkeypatch.setattr(fits, "fit_regression_predictions", fake)
    return calls


def test_q_raw_sorts_each_row_and_floors_at_zero():
    np.testing.assert_array_equal(fits.q_raw([[3.0, 1.0, -2.0], [0.5, -0.1, 0.2]]),
                                  [[0.0, 1.0, 3.0], [0.0, 0.2, 0.5]])


def test_recipe_for_binds_fixed_grid_quantile_loss_and_signed_domain():
    recipe = fits.recipe_for(SPEC, _fit_sealed(), (7, 300), 0.25)
    params = recipe["estimator"]["params"]
    for key, value in SPEC["grid"]["fixed"].items():
        assert params[key] == value, key
    assert (params["max_leaf_nodes"], params["min_samples_leaf"], params["loss"], params["quantile"]) == (
        7, 300, "quantile", 0.25)
    assert recipe["target_domain"] == "signed"
    assert recipe["prediction_projection"] == "none"
    assert recipe["features"] == ["f_a", "f_b", "f_empty"]


def test_arm_id_differs_by_configuration_and_source():
    fixed = SPEC["grid"]["fixed"]
    arm = fits.arm_id("Q_raw", LEVELS, (7, 300), fixed)
    assert arm != fits.arm_id("Q_raw", LEVELS, (15, 300), fixed)
    assert arm != fits.arm_id("Q", LEVELS, (7, 300), fixed)


def test_no_earlier_outputs_returns_first_configuration():
    assert fits.select_configuration(SPEC, [], {}) == ((7, 300), {})


def test_no_earlier_outputs_with_labels_refuses():
    with pytest.raises(OpsError) as exc:
        fits.select_configuration(SPEC, [], {"e1": 0.0})
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", "selection_inputs_invalid")


def test_earlier_outputs_without_any_label_refuse():
    receipt = _receipt(["e1"], {k: [1.0] for k in KEYS})
    for labels in ({}, {"zz": 1.0}):
        with pytest.raises(OpsError) as exc:
            fits.select_configuration(SPEC, [receipt], labels)
        assert _refusal(exc) == ("INVALID_EXPERIMENT_SPEC", "selection_labels_empty")


def test_selection_picks_event_weighted_loss_not_year_average():
    # Constant raw rows at distance d from y=0 cost 0.5 * d at the mean level (the seven levels average 0.5).
    one_year = _receipt(["e1"], {"7x300": [2.0], "7x1000": [2.0], "15x300": [2.0], "15x1000": [0.0]})
    three = _receipt(["e2", "e3", "e4"], {"7x300": [0.0] * 3, "7x1000": [0.8] * 3, "15x300": [0.8] * 3,
                                          "15x1000": [0.8] * 3})
    labels = {e: 0.0 for e in ("e1", "e2", "e3", "e4")}
    winner, losses = fits.select_configuration(SPEC, [one_year, three], labels)
    # Per-year averaging would give 15x1000 (0.2 vs 0.5 for 7x300); event weighting gives 7x300.
    assert losses == pytest.approx({"7x300": 0.25, "7x1000": 0.55, "15x300": 0.55, "15x1000": 0.3})
    assert winner == (7, 300)


def test_exact_tie_prefers_fewer_leaves_then_larger_leaf():
    receipt = _receipt(["e1"], {k: [1.0] for k in KEYS})
    assert fits.select_configuration(SPEC, [receipt], {"e1": 0.0})[0] == (7, 1000)
    two_configs = _spec([7, 15], [300])
    assert fits.select_configuration(two_configs, [receipt], {"e1": 0.0})[0] == (7, 300)


def test_selection_sorts_rows_but_does_not_floor_them():
    unsorted = [0.5, -1.0, 0.0, 0.5, 0.0, 0.5, -1.0]
    receipt = {"event_ids": ["e1"],
               "raw": {"7x300": [unsorted], "15x1000": [sorted(unsorted)], "7x1000": [[-2.0] * 7],
                       "15x300": [[0.0] * 7]}}
    _, losses = fits.select_configuration(SPEC, [receipt], {"e1": 0.0})
    assert losses["7x300"] == pytest.approx(losses["15x1000"])  # both scored on the sorted row
    assert losses["7x1000"] == pytest.approx(1.0)  # -2 stays -2: a 2-unit overshoot, not floored to 0


def test_selection_ignores_events_without_a_label():
    receipt = {"event_ids": ["e1", "e2"],
               "raw": {"7x300": [[0.0] * 7, [50.0] * 7], "7x1000": [[0.0] * 7, [50.0] * 7],
                       "15x300": [[0.0] * 7, [-50.0] * 7], "15x1000": [[0.0] * 7, [-50.0] * 7]}}
    _, losses = fits.select_configuration(SPEC, [receipt], {"e1": 0.0})
    assert losses == pytest.approx({k: 0.0 for k in KEYS})


def test_fit_fold_real_sklearn_shapes_and_determinism():
    sealed = _fit_sealed()
    out = fits.fit_fold(SPEC, sealed, TRAINING, EVALUATION)
    assert out["event_ids"] == EVALUATION
    assert set(out["raw"]) == set(KEYS)
    for rows in out["raw"].values():
        assert np.asarray(rows).shape == (len(EVALUATION), 7)
        assert np.isfinite(np.asarray(rows)).all()
    assert fits.fit_fold(SPEC, sealed, TRAINING, EVALUATION) == out


def test_fit_fold_passes_v14_columns_and_nan_for_none(monkeypatch):
    calls = _patch_fit(monkeypatch)
    fits.fit_fold(SPEC, _fit_sealed(), TRAINING, EVALUATION)
    assert len(calls) == 28
    first = calls[0]
    assert first["columns"] == ["f_a", "f_b", "f_empty"]
    assert first["train_x"].shape == (len(TRAINING), 3)
    f04 = [e for e, _ in TRAINING].index("f04")
    assert np.isnan(first["train_x"][f04, 1])
    assert np.isnan(first["train_x"][:, 2]).all()


def test_fit_fold_keeps_negative_raw_outputs_unclipped(monkeypatch):
    _patch_fit(monkeypatch, fit=lambda recipe, n: {"predictions": [-recipe["estimator"]["params"]["quantile"]] * n})
    out = fits.fit_fold(SPEC, _fit_sealed(), TRAINING, EVALUATION)
    assert out["raw"]["7x300"][0] == pytest.approx([-q for q in LEVELS])


def test_fit_value_error_propagates_unchanged(monkeypatch):
    err = ValueError("fit diverged")

    def broken(recipe, n):
        raise err

    _patch_fit(monkeypatch, fit=broken)
    with pytest.raises(ValueError) as exc:
        fits.fit_fold(SPEC, _fit_sealed(), TRAINING, EVALUATION)
    assert exc.value is err


def test_memory_error_refuses_resource_limit(monkeypatch):
    def out_of_memory(recipe, n):
        raise MemoryError

    _patch_fit(monkeypatch, fit=out_of_memory)
    with pytest.raises(OpsError) as exc:
        fits.fit_fold(SPEC, _fit_sealed(), TRAINING, EVALUATION)
    assert exc.value.problem.code == "RESOURCE_LIMIT_EXCEEDED"


@pytest.mark.parametrize("kind, reason", [
    ("missing_row", "panel_row_missing"),
    ("duplicate_row", "panel_row_duplicate"),
    ("missing_column", "panel_column_missing"),
    ("infinite", "feature_infinite"),
    ("non_numeric", "feature_invalid"),
])
def test_bad_panel_rows_refuse_before_any_fit(monkeypatch, kind, reason):
    sealed = _fit_sealed()
    target = next(r for r in sealed["rows"] if r["event_id"] == "f00")
    if kind == "missing_row":
        sealed["rows"].remove(target)
    elif kind == "duplicate_row":
        sealed["rows"].append(dict(target))
    elif kind == "missing_column":
        del target["f_b"]
    elif kind == "infinite":
        target["f_a"] = float("inf")
    else:
        target["f_a"] = "x"
    calls = _patch_fit(monkeypatch)
    with pytest.raises(OpsError) as exc:
        fits.fit_fold(SPEC, sealed, TRAINING, EVALUATION)
    assert _refusal(exc) == ("EXPERIMENT_IDENTITY_CONFLICT", reason)
    assert calls == []


def test_fit_fold_real_sklearn_recovers_a_step_function():
    # Noise-free two-value target: every level of the quantile fit should sit on its side's value.
    spec = _spec([7], [20])
    rows = [{"event_id": f"s{k:03d}", "f_a": -1.0 if k % 2 == 0 else 1.0, "f_b": 0.0, "f_empty": None}
            for k in range(400)]
    sealed = {"model_columns": ["f_a", "f_b", "f_empty"], "dictionary_hash": "sha256:" + "a" * 64, "rows": rows}
    training = [(r["event_id"], 2.0 if r["f_a"] < 0 else 10.0) for r in rows[:360]]
    evaluation = [r["event_id"] for r in rows[360:]]
    side = {r["event_id"]: (2.0 if r["f_a"] < 0 else 10.0) for r in rows[360:]}
    out = fits.fit_fold(spec, sealed, training, evaluation)
    assert out["event_ids"] == evaluation
    assert len(out["raw"]) == 1
    for raw in out["raw"].values():
        raw = np.asarray(raw)
        assert raw.shape == (len(evaluation), 7)
        for i, event_id in enumerate(evaluation):
            assert np.abs(raw[i] - side[event_id]).max() <= 0.2, event_id
