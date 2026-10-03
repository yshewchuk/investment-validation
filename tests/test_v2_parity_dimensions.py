"""Unit tests for ``engine.v2.parity.dimensions.compare_dimension``.

The v1.2 widening adds ``"values"`` -- each disagreeing field's raw
legacy/native pair -- alongside the existing ``"agree"``/``"finding_fields"``/
``"receipt"`` keys.  These call the one function directly; the Phase 4
checker's wrapper is the negative control proving the addition changed
nothing for that existing caller.
"""
from __future__ import annotations

from checks import phase4_real
from engine.v2.parity.dimensions import compare_dimension


def test_agreeing_dimension_has_no_values():
    result = compare_dimension({"gate_score": 0.5}, {"gate_score": 0.5}, "verdicts")

    assert result["agree"] is True
    assert result["finding_fields"] == []
    assert result["values"] == {}


def test_differing_field_carries_both_raw_values():
    result = compare_dimension(
        {"gate_score": 0.5, "gate_pass": True},
        {"gate_score": 0.75, "gate_pass": True},
        "verdicts")

    assert result["agree"] is False
    assert result["finding_fields"] == ["gate_score"]
    assert result["values"] == {"gate_score": {"legacy": 0.5, "native": 0.75}}


def test_phase4_wrapper_behavior_is_unchanged():
    agree = phase4_real._compare_dimension(
        {"exp_pnl_model": 1.0}, {"exp_pnl_model": 1.0}, "simulation")
    assert agree["agree"] is True
    assert agree["finding_fields"] == []

    differ = phase4_real._compare_dimension(
        {"exp_pnl_model": 1.0}, {"exp_pnl_model": 1.5}, "simulation")
    assert differ["agree"] is False
    assert differ["finding_fields"] == ["exp_pnl_model"]
