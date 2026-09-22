"""Regression test for the THIN_ANALOGS gap: ``native_analog`` correctly
computes ``AnalogResult.thin`` (engine/v2/scoring/native_analog.py:536,
:562), but ``_execute_analogs`` and ``_execute_frozen_analogs``
(engine/v2/scoring/stages.py) built their output dicts without ever
reading it, so THIN_ANALOGS never reached the flags list natively even
though legacy raises it (engine/analogs.py:1011, engine/score.py:3168).
"""
from __future__ import annotations

from types import SimpleNamespace

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
