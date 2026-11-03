"""Focused tests for the DYN-SV chooser wiring in native_score_batch."""
from __future__ import annotations

import pickle
from dataclasses import make_dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.ops import native_score_batch as nsb
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
from engine.v2.scoring.release_bindings import ModelNotReady

Bundle = make_dataclass(
    "Bundle",
    ["model_release", "chooser_recipe", "chooser_fold_pools",
     "chooser_analog_pool", "chooser_admissible_table"],
    frozen=True,
)
Release = make_dataclass("Release", ["bindings"], frozen=True)
Binding = make_dataclass("Binding", ["binding_id"], frozen=True)

_KEY = BoardRequest("AAA", "TWIN-P", pd.Timestamp("2026-01-12"), "close")
_CHOOSER_NAMES = {
    "TWIN-P", "TWIN-P5", "BFLY-P", "BFLY-P5", "CTR5", "RAMP7", "CND-PS",
}


def _bundle():
    return Bundle(Release((Binding("existing"),)), {}, {}, None, None)


def _state():
    return SimpleNamespace(
        producer_bindings={
            "pred_im_t1_d14": Binding("producer-im"),
            "pred_runup_abs_move_d14": Binding("producer-ru"),
        },
        fold_pools={"pred_abs_move": {
            "predictions": (1.0,), "residuals": (0.0,), "interval_floor": 0.1}},
        analog_pool=SimpleNamespace(
            pool_id="pool", cutoff="2026-01-10", content_hash="hash-pool"),
        admissible_table=SimpleNamespace(
            table_id="table", version="v1", content_hash="hash-table"),
    )


def _binding():
    return SimpleNamespace(
        model_identity={"chooser:DYN-SV": Binding("chooser")})


def test_only_the_seven_menu_specs_choose():
    for name, spec in nsb._STRATEGY_REGISTRY.items():
        assert spec.chooser is (name in _CHOOSER_NAMES)
    assert nsb._STRATEGY_REGISTRY["STR-THRU"].chooser is False
    assert nsb._STRATEGY_REGISTRY["STR-RUNUP"].chooser is False


def test_selected_state_declares_recipe_and_adds_missing_producers(monkeypatch):
    state = _state()
    state.producer_bindings["pred_im_t1_d14"] = Binding("producer-im")
    bundle = Bundle(
        Release((Binding("existing"), Binding("producer-im"))), {}, {}, None, None)
    captured = {}

    def fake(binding, event_date, as_of, *, fold_cache):
        captured.update(
            binding=binding, event_date=event_date, as_of=as_of, fold_cache=fold_cache)
        return state

    monkeypatch.setattr(nsb, "select_chooser_state", fake)
    cache = {}
    result = nsb._chooser_bundle_or_refusal(
        bundle, _KEY, binding=_binding(), event_date="2026-01-12",
        as_of="2026-01-10", fold_cache=cache)

    assert not isinstance(result, NativeScoreBatchRowRefusal)
    assert captured["fold_cache"] is cache
    assert result.model_release.bindings == (
        Binding("existing"), Binding("producer-im"), Binding("producer-ru"))
    assert bundle.model_release.bindings == (
        Binding("existing"), Binding("producer-im"))
    assert result.chooser_recipe == {
        "binding_id": "chooser",
        "producers": {
            "pred_im_t1_d14": {"binding_id": "producer-im"},
            "pred_runup_abs_move_d14": {"binding_id": "producer-ru"},
        },
        "analog_pool": {
            "pool_id": "pool", "cutoff": "2026-01-10", "content_hash": "hash-pool"},
        "admissible_table": {
            "table_id": "table", "version": "v1", "content_hash": "hash-table"},
    }
    assert result.chooser_fold_pools is state.fold_pools
    assert result.chooser_analog_pool is state.analog_pool
    assert result.chooser_admissible_table is state.admissible_table


def test_model_not_ready_becomes_the_fixed_row_refusal(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ModelNotReady("chooser_analog_pool", "no unique analog pool for cutoff")

    monkeypatch.setattr(nsb, "select_chooser_state", refuse)
    result = nsb._chooser_bundle_or_refusal(
        _bundle(), _KEY, binding=_binding(), event_date="2026-01-12",
        as_of="2026-01-10", fold_cache={})

    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert (result.code, result.detail) == (
        "MODEL_NOT_READY", "DYN-SV chooser state missing or unusable")


@pytest.mark.parametrize("name", ["STR-THRU", "STR-RUNUP"])
def test_non_chooser_spec_returns_the_same_bundle(monkeypatch, name):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("non-chooser spec must not select chooser state")

    monkeypatch.setattr(nsb, "select_chooser_state", unexpected)
    bundle = _bundle()
    before = pickle.dumps(bundle)
    result = nsb._apply_declared_chooser(
        bundle, _KEY, nsb._STRATEGY_REGISTRY[name],
        SimpleNamespace(calendar_row={"event_date": "2026-01-12"}),
        binding=object(), as_of="2026-01-10", fold_cache={})

    assert result is bundle
    assert pickle.dumps(result) == before
