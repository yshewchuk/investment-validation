"""Regression test for the NO_PAYOFF_MAP gap: legacy's site-1 check
(engine/score.py:2799, ``PAYOFF_DRIVER.get(strategy) is None``) fires from
the strategy's identity alone, before any output-anomaly check. Native's
``_execute_model`` (engine/v2/scoring/stages.py) early-returned ``{}`` for
a row with no ``payoff_recipe`` at all -- which is exactly what capture
records for a strategy with no payoff driver, since legacy's own model
stage returns immediately too -- without ever reaching
``_model_driver_and_cost``, where the equivalent check already lived. That
silently dropped NO_PAYOFF_MAP for every strategy absent from
PAYOFF_DRIVER (10 rows).

``checks/phase5_consumers.py``'s ``_row`` helper bundled NO_PAYOFF_MAP into
its generic "unresolved" set; that is correct for its STR-THRU/STR-RUNUP
callers but was incidentally masking this exact defect for the TWIN-P
chooser probe (TWIN-P has no payoff driver and always carries
NO_PAYOFF_MAP in legacy, by design -- see engine/payoff.py's PAYOFF_DRIVER
docstring). Fixed alongside this gap with a dedicated ``_chooser_row`` that
does not treat NO_PAYOFF_MAP as unresolved for that probe.
"""
from __future__ import annotations

from types import SimpleNamespace

from engine.v2.scoring.stages import _execute_model


def test_no_payoff_map_fires_for_unsupported_strategy_with_no_recipe():
    inputs = SimpleNamespace(model={})  # no payoff_recipe declared at all
    values: dict = {}
    flags: list = []
    output = _execute_model(inputs, "CAL-P", values, flags)

    assert output == {}
    assert "NO_PAYOFF_MAP" in flags
    assert "UNOWNED_MODEL_OUTPUT" not in flags


def test_no_payoff_map_does_not_fire_for_a_supported_strategy_with_no_recipe():
    """The row genuinely never asked for a model calculation; silence is
    correct for STR-THRU/STR-RUNUP absent a recipe, exactly as before."""
    inputs = SimpleNamespace(model={})
    values: dict = {}
    flags: list = []
    output = _execute_model(inputs, "STR-THRU", values, flags)

    assert output == {}
    assert "NO_PAYOFF_MAP" not in flags


def test_unowned_model_output_still_fires_for_a_supported_strategy_with_stray_outputs():
    """UNOWNED_MODEL_OUTPUT behavior (the pre-existing branch) must be
    unchanged for a supported strategy whose block carries an output with
    no recipe behind it."""
    inputs = SimpleNamespace(model={"exp_pnl_model": 0.1})
    values: dict = {}
    flags: list = []
    output = _execute_model(inputs, "STR-THRU", values, flags)

    assert output == {}
    assert "UNOWNED_MODEL_OUTPUT" in flags
    assert "NO_PAYOFF_MAP" not in flags


def test_no_payoff_map_takes_precedence_over_unowned_model_output_for_unsupported_strategy():
    """Legacy's site-1 check fires from strategy identity ALONE, before any
    output-anomaly check -- a stray output on an unsupported strategy must
    not mask NO_PAYOFF_MAP with UNOWNED_MODEL_OUTPUT instead."""
    inputs = SimpleNamespace(model={"exp_pnl_model": 0.1})
    values: dict = {}
    flags: list = []
    output = _execute_model(inputs, "CAL-P", values, flags)

    assert output == {}
    assert "NO_PAYOFF_MAP" in flags
    assert "UNOWNED_MODEL_OUTPUT" not in flags
