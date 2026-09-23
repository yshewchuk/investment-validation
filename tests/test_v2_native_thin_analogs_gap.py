"""Regression test for the THIN_ANALOGS gap: ``native_analog`` correctly
computes ``AnalogResult.thin`` (engine/v2/scoring/native_analog.py:536,
:562), but ``_execute_analogs`` and ``_execute_frozen_analogs``
(engine/v2/scoring/stages.py) built their output dicts without ever
reading it, so THIN_ANALOGS never reached the flags list natively even
though legacy raises it (engine/analogs.py:1011, engine/score.py:3168).
"""
from __future__ import annotations

from types import SimpleNamespace

from engine.v2.scoring.native_analog import (
    LEGACY_BUCKET_DIMENSIONS,
    LEGACY_WIDENING_ORDER,
    LegacyBucketRecipe,
    bucket_population_hash,
    legacy_bucket_bootstrap_seed,
)
from engine.v2.scoring.stages import _execute_analogs, _execute_frozen_analogs


class _FakeAnalogResult(SimpleNamespace):
    pass


def _fake_result(thin: bool) -> _FakeAnalogResult:
    return _FakeAnalogResult(
        exp_pnl_analog=0.1, win_analog=0.5, ci_low=None, ci_high=None,
        n_analogs=2, thin=thin,
    )


def test_execute_analogs_emits_thin_analogs_when_result_is_thin(monkeypatch):
    import engine.v2.scoring.native_analog as native_analog

    monkeypatch.setattr(
        native_analog, "evaluate_analogs",
        lambda **kwargs: _fake_result(thin=True),
    )
    inputs = SimpleNamespace(analogs={
        "recipe": {"feature_names": ("x",), "neighbors": 1,
                   "population_hash": "sha256:" + "0" * 64},
        "source_rows": [{"row_id": "a", "features": {"x": 0.0},
                          "realized_pnl": 1.0}],
        "query_features": {"x": 0.0},
    })
    values: dict = {}
    flags: list = []
    output = _execute_analogs(inputs, values, flags, strategy="STR-THRU")

    assert "THIN_ANALOGS" in flags
    assert output["n_analogs"] == 2


def test_execute_analogs_does_not_emit_thin_analogs_when_result_is_not_thin(monkeypatch):
    import engine.v2.scoring.native_analog as native_analog

    monkeypatch.setattr(
        native_analog, "evaluate_analogs",
        lambda **kwargs: _fake_result(thin=False),
    )
    inputs = SimpleNamespace(analogs={
        "recipe": {"feature_names": ("x",), "neighbors": 1,
                   "population_hash": "sha256:" + "0" * 64},
        "source_rows": [{"row_id": "a", "features": {"x": 0.0},
                          "realized_pnl": 1.0}],
        "query_features": {"x": 0.0},
    })
    values: dict = {}
    flags: list = []
    _execute_analogs(inputs, values, flags, strategy="STR-THRU")

    assert "THIN_ANALOGS" not in flags


def test_execute_frozen_analogs_emits_thin_analogs_when_result_is_thin(monkeypatch):
    import engine.v2.scoring.native_analog as native_analog

    monkeypatch.setattr(
        native_analog, "evaluate_frozen_analogs",
        lambda **kwargs: (_fake_result(thin=True), None),
    )
    block = {"analog_artifact": object(), "analog_artifact_recipe": {}}
    values: dict = {"fill": 0.5}
    flags: list = []
    output = _execute_frozen_analogs(block, "STR-THRU", values, flags)

    assert "THIN_ANALOGS" in flags
    assert output["n_analogs"] == 2


def test_execute_frozen_analogs_does_not_emit_thin_analogs_when_not_thin(monkeypatch):
    import engine.v2.scoring.native_analog as native_analog

    monkeypatch.setattr(
        native_analog, "evaluate_frozen_analogs",
        lambda **kwargs: (_fake_result(thin=False), None),
    )
    block = {"analog_artifact": object(), "analog_artifact_recipe": {}}
    values: dict = {"fill": 0.5}
    flags: list = []
    _execute_frozen_analogs(block, "STR-THRU", values, flags)

    assert "THIN_ANALOGS" not in flags


def test_execute_analogs_on_a_real_empty_declared_population_is_thin_with_zero_analogs():
    """End-to-end (no ``evaluate_analogs`` mock): an explicit empty-population
    recipe -- what ``engine.score.Phase4TraceCollector.capture_analog_inputs``
    now writes for a row where legacy's analog matcher ran and found an empty
    causal population, instead of skipping the key -- must drive
    ``_execute_analogs`` to ``n_analogs == 0`` and ``THIN_ANALOGS``, matching
    legacy's own record for that row (``n_analogs`` present, the analog
    numbers ``None``, THIN_ANALOGS set)."""
    query = {
        "mcap_bucket": "1-10B", "moneyness_band": "ATM",
        "dte_band": "4-10", "implied_tercile": "mid",
    }
    recipe = LegacyBucketRecipe(
        bucket_dimensions=LEGACY_BUCKET_DIMENSIONS,
        widening_order=LEGACY_WIDENING_ORDER,
        min_analogs=30,
        alpha=0.5,
        bootstrap_draws=2000,
        bootstrap_seed=legacy_bucket_bootstrap_seed(
            snapshot="snap", strategy="STR-THRU", alpha=0.5,
            buckets=query, request_key="req",
        ),
        ci_quantiles=(0.05, 0.95),
        population_hash=bucket_population_hash([], LEGACY_BUCKET_DIMENSIONS),
    )
    inputs = SimpleNamespace(analogs={
        "recipe": vars(recipe),
        "source_rows": [],
        "query_features": query,
    })
    values: dict = {}
    flags: list = []
    output = _execute_analogs(inputs, values, flags, strategy="STR-THRU")

    assert output["n_analogs"] == 0
    assert output["exp_pnl_analog"] is None
    assert output["win_analog"] is None
    assert output["ci_low"] is None
    assert output["ci_high"] is None
    assert "THIN_ANALOGS" in flags
    assert values["n_analogs"] == 0


def test_execute_frozen_analogs_still_returns_early_on_model_not_ready(monkeypatch):
    """A refusal (MODEL_NOT_READY etc.) must still short-circuit before any
    thin check -- there is no ``result`` to read ``.thin`` from."""
    import engine.v2.scoring.native_analog as native_analog

    monkeypatch.setattr(
        native_analog, "evaluate_frozen_analogs",
        lambda **kwargs: (None, "MODEL_NOT_READY"),
    )
    block = {"analog_artifact": None, "analog_artifact_recipe": {}}
    values: dict = {"fill": 0.5}
    flags: list = []
    output = _execute_frozen_analogs(block, "STR-THRU", values, flags)

    assert output == {}
    assert flags == ["MODEL_NOT_READY"]
