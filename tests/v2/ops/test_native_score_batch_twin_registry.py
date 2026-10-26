"""Tests for native_score_batch TWIN registry, spec guards, decision clock,
mcap/pre_iv30 guards and entry-rule declarations (synthetic only)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from engine.v2.ops import native_score_batch as nsb
from engine.v2.ops.native_score_batch import (
    _DriverSpec, _StrategySpec, _STRATEGY_REGISTRY, _matched_decision_clock,
    _mcap_usd, _pre_iv30, _entry_rule_declarations_or_refusal)


def _identity(model_id, clock="clock", role=None):
    return SimpleNamespace(model_id=model_id, decision_clock_id=clock, role=role)


def _twin_identities(size_clock="clock", crush_clock="clock"):
    return {"size:*": _identity("m-size", size_clock, "size:*"),
            "iv_crush:*": _identity("m-crush", crush_clock, "iv_crush:*")}


def test_registry_has_gateless_twin_specs():
    for name in ("TWIN-P", "TWIN-P5"):
        spec = _STRATEGY_REGISTRY[name]
        assert (spec.gate_role, spec.gate_policy_key, spec.entry_rule) == (None, None, True)
        assert [(d.output, d.role) for d in spec.drivers] == [
            ("forecast_abs_move", "size:*"), ("pred_iv_crush_30", "iv_crush:*")]
    for name in ("STR-THRU", "STR-RUNUP"):
        spec = _STRATEGY_REGISTRY[name]
        assert spec.gate_role is not None and spec.gate_policy_key is not None


@pytest.mark.parametrize("kwargs,match", [
    (dict(drivers=(_DriverSpec("forecast_abs_move", "size:*"),),
          bundle_strategy="X", gate_role="gate:X", gate_policy_key=None),
     "both None or both set"),
    (dict(drivers=(_DriverSpec("forecast_abs_move", "size:*"),
                   _DriverSpec("pred_iv_crush_30", "iv_crush:*")),
          bundle_strategy="X", entry_rule=True, gate_role="gate:X",
          gate_policy_key="X"), "must be gate-less"),
    (dict(drivers=(_DriverSpec("forecast_abs_move", "size:*"),
                   _DriverSpec("wrong", "iv_crush:*")),
          bundle_strategy="X", entry_rule=True),
     "forecast_abs_move then pred_iv_crush_30")])
def test_strategy_spec_rejects(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _StrategySpec(**kwargs)


def test_clock_happy_path():
    code, detail, pair = _matched_decision_clock(_twin_identities(), _STRATEGY_REGISTRY["TWIN-P"])
    assert code is None and detail is None
    drivers, gate = pair
    assert gate is None
    assert [d.output for d in drivers] == ["forecast_abs_move", "pred_iv_crush_30"]


def test_clock_missing_crush_role():
    ids = _twin_identities()
    del ids["iv_crush:*"]
    assert _matched_decision_clock(ids, _STRATEGY_REGISTRY["TWIN-P"]) == (
        "RELEASE_MISSING_ROLE", "iv_crush:*", None)


def test_clock_ambiguous_decision_clock():
    code, detail, pair = _matched_decision_clock(
        _twin_identities("a", "b"), _STRATEGY_REGISTRY["TWIN-P"])
    assert code == "AMBIGUOUS_DECISION_CLOCK" and pair is None and "iv_crush:*" in detail


@pytest.mark.parametrize("value,expected", [
    (25.0, math.exp(25.0)), (None, None), (float("nan"), None),
    (float("inf"), None), (True, None), ("25", None), (1e9, None)])
def test_mcap_usd(value, expected):
    assert _mcap_usd({"mcap_log": value}) == expected


@pytest.mark.parametrize("value,expected", [
    (45.0, 45.0), (0, None), (-1, None), (float("nan"), None),
    (None, None), ("45", None), (True, None)])
def test_pre_iv30(value, expected):
    assert _pre_iv30({"pre_iv30": value}) == expected


def test_entry_rule_declarations_or_refusal(monkeypatch):
    captured = {}
    def fake(binding, **kwargs):
        captured.update(kwargs)
        return {"paired_residual_artifact": "pool",
                "paired_residual_recipe": {"cutoff": "2026-01-09"},
                "trailing_cutoff_artifact": "bar",
                "gate_recipe": {"mode": "entry_rule"}}
    monkeypatch.setattr(nsb, "entry_rule_declarations", fake)
    size = _identity("m-size", "clock", "size:*")
    crush = _identity("m-crush", "clock", "iv_crush:*")
    binding = SimpleNamespace()
    row = {"event_date": "2026-01-12", "entry_date": "2026-01-09"}
    call = lambda **panel: _entry_rule_declarations_or_refusal(
        binding, strategy="TWIN-P", entry_rule_ids=(size, crush),
        as_of="2026-01-10", calendar_row=row, panel_row=panel)
    no_iv = call(mcap_log=25.0)
    assert no_iv["residual_recipe"] == {"mode": "planned_exit"}
    assert "pre_iv30" not in no_iv["paired_residual_recipe"]
    assert captured["mcap_usd"] == math.exp(25.0)
    with_iv = call(mcap_log=25.0, pre_iv30=45.0)
    assert with_iv["paired_residual_recipe"] == {
        "cutoff": "2026-01-09", "pre_iv30": 45.0}
