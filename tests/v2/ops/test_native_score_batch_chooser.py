"""Focused tests for the DYN-SV chooser wiring in native_score_batch."""
from __future__ import annotations

import pickle
from dataclasses import field, make_dataclass
from types import SimpleNamespace

import pandas as pd
import pytest

from engine.v2.ops import native_score_batch as nsb
from engine.v2.ops.native_board_universe import BoardRequest
from engine.v2.ops.native_score_batch import (
    NativeScoreBatchRowRefusal,
    NightlyEventInputs,
)
from engine.v2.scoring.release_bindings import ModelNotReady

Bundle = make_dataclass(
    "Bundle",
    ["model_release", "chooser_recipe", "chooser_fold_pools",
     "chooser_analog_pool", "chooser_admissible_table",
     ("gate_forecast_source", object, field(default=None))],
    frozen=True,
)
Release = make_dataclass("Release", ["bindings"], frozen=True)
Member = make_dataclass("Member", ["name", "content_hash"], frozen=True)
Binding = make_dataclass(
    "Binding", ["binding_id", ("members", tuple, field(default=()))],
    frozen=True)

_CHOOSER_MEMBER = (Member("estimator", "hash-est-chooser"),)
_IM_MEMBER = (Member("estimator", "hash-est-im"),)
_RU_MEMBER = (Member("estimator", "hash-est-ru"),)

_KEY = BoardRequest("AAA", "TWIN-P", pd.Timestamp("2026-01-12"), "close")
_CHOOSER_NAMES = {
    "TWIN-P", "TWIN-P5", "BFLY-P", "BFLY-P5", "CTR5", "RAMP7", "CND-PS",
}
_CHOOSER_PRIMITIVES = (
    "mean_prior_abs_move", "ema12r_abs", "signed_streak",
    "mean_prior_or_implied", "mcap_log", "or_implied", "or_rvol30",
    "spy_ret21", "spy_ret63", "spy_ret252", "spy_dd252", "spy_vol20",
    "spy_vol5", "spy_vol60", "spy_vol252", "spy_vol20_rel252",
)


def _bundle():
    return Bundle(Release((Binding("existing"),)), {}, {}, None, None)


def _state():
    return SimpleNamespace(
        producer_bindings={
            "pred_im_t1_d14": Binding("producer-im", _IM_MEMBER),
            "pred_runup_abs_move_d14": Binding("producer-ru", _RU_MEMBER),
        },
        fold_pools={"pred_abs_move": {
            "predictions": (1.0,), "residuals": (0.0,), "interval_floor": 0.1}},
        analog_pool=SimpleNamespace(
            pool_id="pool", cutoff="2026-01-10", content_hash="hash-pool"),
        admissible_table=SimpleNamespace(
            table_id="table", version="v1", content_hash="hash-table"),
        artifact_refs=("hash-state-1", "hash-pool"),
        fold_provenance={
            "pred_im_t1_d14": {
                "training_cutoff": "2025-12-01", "fold_start": "2025-12-15",
                "descriptor_hash": "hash-desc-im"},
            "pred_runup_abs_move_d14": {
                "training_cutoff": "2025-12-01", "fold_start": "2025-12-15",
                "descriptor_hash": "hash-desc-ru"},
        },
    )


def _binding():
    return SimpleNamespace(
        model_identity={"chooser:DYN-SV": Binding("chooser", _CHOOSER_MEMBER)})


def _projection_event(strategy: str) -> NightlyEventInputs:
    panel = {name: 0.1 for name in _CHOOSER_PRIMITIVES}
    panel["zeta_distinct"] = 0.5
    panel["eta_distinct"] = 0.5
    return NightlyEventInputs(
        key=BoardRequest("AAA", strategy, pd.Timestamp("2026-01-12"), "close"),
        calendar_row={"event_date": "2026-01-12", "entry_date": "2026-01-12"},
        panel_row=panel, panel_anchor="2026-01-10",
        tier4_row={}, quote_rows=())


def test_only_the_seven_menu_specs_choose():
    assert _CHOOSER_NAMES <= set(nsb._STRATEGY_REGISTRY)
    for name, spec in nsb._STRATEGY_REGISTRY.items():
        assert spec.chooser is (name in _CHOOSER_NAMES)
    assert nsb._STRATEGY_REGISTRY["STR-THRU"].chooser is False
    assert nsb._STRATEGY_REGISTRY["STR-RUNUP"].chooser is False


def test_selected_state_declares_recipe_and_adds_missing_producers(monkeypatch):
    state = _state()
    binding = _binding()
    bundle = Bundle(
        Release((Binding("chooser", _CHOOSER_MEMBER), Binding("existing"),
                 Binding("producer-im", _IM_MEMBER))),
        {}, {}, None, None)
    captured = {}

    def fake(binding, event_date, as_of):
        captured.update(
            binding=binding, event_date=event_date, as_of=as_of)
        return state

    monkeypatch.setattr(nsb, "select_chooser_state", fake)
    result = nsb._chooser_bundle_or_refusal(
        bundle, _KEY, binding=binding, event_date="2026-01-12",
        as_of="2026-01-10")

    assert not isinstance(result, NativeScoreBatchRowRefusal)
    assert captured == {
        "binding": binding, "event_date": "2026-01-12", "as_of": "2026-01-10"}
    assert result.model_release.bindings == (
        Binding("chooser", _CHOOSER_MEMBER), Binding("existing"),
        Binding("producer-im", _IM_MEMBER), Binding("producer-ru", _RU_MEMBER))
    assert bundle.model_release.bindings == (
        Binding("chooser", _CHOOSER_MEMBER), Binding("existing"),
        Binding("producer-im", _IM_MEMBER))
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
        "artifact_refs": ["hash-state-1", "hash-pool"],
        "fold_provenance": {
            "pred_im_t1_d14": {
                "training_cutoff": "2025-12-01", "fold_start": "2025-12-15",
                "descriptor_hash": "hash-desc-im"},
            "pred_runup_abs_move_d14": {
                "training_cutoff": "2025-12-01", "fold_start": "2025-12-15",
                "descriptor_hash": "hash-desc-ru"},
        },
    }
    assert result.chooser_fold_pools is state.fold_pools
    assert result.chooser_analog_pool is state.analog_pool
    assert result.chooser_admissible_table is state.admissible_table

    drivers = (SimpleNamespace(identity=SimpleNamespace(artifact_hash="hash-driver")),)
    refs = nsb._score_artifact_refs(drivers, None, result)
    assert refs == (
        "hash-driver",
        "hash-est-chooser", "hash-est-im", "hash-est-ru",
        "hash-pool", "hash-table", "hash-state-1",
    )
    assert len(refs) == len(set(refs))


def test_model_not_ready_becomes_the_fixed_row_refusal(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise ModelNotReady("chooser_analog_pool", "no unique analog pool for cutoff")

    monkeypatch.setattr(nsb, "select_chooser_state", refuse)
    result = nsb._chooser_bundle_or_refusal(
        _bundle(), _KEY, binding=_binding(), event_date="2026-01-12",
        as_of="2026-01-10")

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
        binding=object(), as_of="2026-01-10")

    assert result is bundle
    assert pickle.dumps(result) == before


def test_project_member_features_projects_selected_state_order(monkeypatch):
    state = SimpleNamespace(producer_bindings={
        "pred_im_t1_d14": SimpleNamespace(
            feature_order=("zeta_distinct", "entry_cost_pct", "days_to_print",
                           "dte_entry")),
        "pred_runup_abs_move_d14": SimpleNamespace(
            feature_order=("eta_distinct", "pred_abs_move", "analog_mean")),
    })
    captured = {}

    def fake(binding, event_date, as_of):
        captured.update(binding=binding, event_date=event_date, as_of=as_of)
        return state

    monkeypatch.setattr(nsb, "select_chooser_state", fake)
    event = _projection_event("TWIN-P")
    binding = _binding()
    projected = nsb._project_member_features(
        event.key, event, binding, ("signal",), as_of="2026-01-10")

    assert captured == {"binding": binding, "event_date": "2026-01-12",
                        "as_of": "2026-01-10"}
    expected = set(_CHOOSER_PRIMITIVES) | {
        "zeta_distinct", "eta_distinct", "signal"}
    assert set(projected) == expected
    assert tuple(projected) == tuple(sorted(projected))
    for excluded in ("entry_cost_pct", "days_to_print", "dte_entry",
                     "pred_abs_move", "analog_mean"):
        assert excluded not in projected

    def unexpected(*_args, **_kwargs):
        raise AssertionError("non-chooser strategy must not select chooser state")

    monkeypatch.setattr(nsb, "select_chooser_state", unexpected)
    kept = ("signal", "pred_abs_move")
    thru = _projection_event("STR-THRU")
    unchanged = nsb._project_member_features(
        thru.key, thru, binding, kept, as_of="2026-01-10")
    assert unchanged is kept


@pytest.mark.parametrize(
    ("producer_feature_order", "code", "detail"),
    [
        (
            (),
            "RELEASE_MISSING_FEATURE_ORDER",
            "chooser producer identity has no feature_order",
        ),
        (
            ("missing_producer_feature",),
            "MISSING_STAGED_INPUT",
            "required member feature is missing from staged rows",
        ),
    ],
)
def test_project_member_features_refuses_bad_producer_contract(
        monkeypatch, producer_feature_order, code, detail):
    state = SimpleNamespace(producer_bindings={
        "pred_im_t1_d14": SimpleNamespace(feature_order=producer_feature_order),
    })
    monkeypatch.setattr(nsb, "select_chooser_state", lambda *a, **k: state)
    event = _projection_event("TWIN-P")

    refusal = nsb._project_member_features(
        event.key, event, _binding(), ("signal",), as_of="2026-01-10")

    assert isinstance(refusal, NativeScoreBatchRowRefusal)
    assert (refusal.code, refusal.detail) == (code, detail)


@pytest.mark.parametrize(
    ("binding_kind", "binding_id"),
    [
        ("chooser", None),
        ("chooser", ""),
        ("producer", None),
        ("producer", ""),
    ],
)
def test_chooser_binding_guard_refuses_missing_producer_binding_id(
        monkeypatch, binding_kind, binding_id):
    state = _state()
    binding = _binding()
    if binding_kind == "chooser":
        binding = SimpleNamespace(model_identity={
            "chooser:DYN-SV": Binding(binding_id, _CHOOSER_MEMBER)})
    else:
        state.producer_bindings["pred_im_t1_d14"] = SimpleNamespace(
            binding_id=binding_id)
    monkeypatch.setattr(nsb, "select_chooser_state", lambda *a, **k: state)

    result = nsb._chooser_bundle_or_refusal(
        _bundle(), _KEY, binding=binding, event_date="2026-01-12",
        as_of="2026-01-10")

    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert (result.code, result.detail) == (
        "MODEL_NOT_READY", "DYN-SV chooser state missing or unusable")


def _state_with_producer_features():
    state = _state()
    state.producer_bindings = {
        "pred_im_t1_d14": SimpleNamespace(
            binding_id="producer-im", feature_order=("zeta_distinct",)),
        "pred_runup_abs_move_d14": SimpleNamespace(
            binding_id="producer-ru", feature_order=("eta_distinct",)),
    }
    return state


def test_one_selection_shared_by_projection_and_declaration(monkeypatch):
    state = _state_with_producer_features()
    calls = []

    def fake(binding, event_date, as_of):
        calls.append((binding, event_date, as_of))
        return state

    monkeypatch.setattr(nsb, "select_chooser_state", fake)
    event = _projection_event("TWIN-P")
    binding = _binding()
    selection = nsb._ChooserSelection()

    projected = nsb._project_member_features(
        event.key, event, binding, ("signal",), as_of="2026-01-10",
        selection=selection)
    assert not isinstance(projected, NativeScoreBatchRowRefusal)
    assert {"zeta_distinct", "eta_distinct", "signal"} <= set(projected)

    result = nsb._chooser_bundle_or_refusal(
        _bundle(), event.key, binding=binding,
        event_date=event.calendar_row.get("event_date"),
        as_of="2026-01-10", selection=selection)

    assert not isinstance(result, NativeScoreBatchRowRefusal)
    assert len(calls) == 1
    assert result.chooser_recipe["binding_id"] == "chooser"
    assert result.chooser_recipe["producers"] == {
        "pred_im_t1_d14": {"binding_id": "producer-im"},
        "pred_runup_abs_move_d14": {"binding_id": "producer-ru"},
    }


def test_one_selection_shared_when_selector_not_ready(monkeypatch):
    calls = []

    def refuse(*_args, **_kwargs):
        calls.append(1)
        raise ModelNotReady("chooser_analog_pool", "no unique analog pool for cutoff")

    monkeypatch.setattr(nsb, "select_chooser_state", refuse)
    event = _projection_event("TWIN-P")
    binding = _binding()
    selection = nsb._ChooserSelection()
    kept = ("signal", "pred_abs_move")

    projected = nsb._project_member_features(
        event.key, event, binding, kept, as_of="2026-01-10",
        selection=selection)
    assert projected is kept

    result = nsb._chooser_bundle_or_refusal(
        _bundle(), event.key, binding=binding,
        event_date=event.calendar_row.get("event_date"),
        as_of="2026-01-10", selection=selection)

    assert isinstance(result, NativeScoreBatchRowRefusal)
    assert (result.code, result.detail) == (
        "MODEL_NOT_READY", "DYN-SV chooser state missing or unusable")
    assert len(calls) == 1


def test_project_member_features_refuses_none_chooser_primitive(monkeypatch):
    state = SimpleNamespace(producer_bindings={})
    monkeypatch.setattr(nsb, "select_chooser_state", lambda *a, **k: state)
    event = _projection_event("TWIN-P")
    event.panel_row["spy_ret21"] = None

    refusal = nsb._project_member_features(
        event.key, event, _binding(), ("signal",), as_of="2026-01-10")

    assert isinstance(refusal, NativeScoreBatchRowRefusal)
    assert (refusal.code, refusal.detail) == (
        "MISSING_STAGED_INPUT",
        "required member feature is missing from staged rows")


def test_null_panel_row_refuses_before_chooser_state_selection(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("malformed staged row must not select chooser state")

    monkeypatch.setattr(nsb, "select_chooser_state", unexpected)
    event = NightlyEventInputs(
        key=_KEY,
        calendar_row={"event_date": "2026-01-12", "entry_date": "2026-01-12"},
        panel_row=None, panel_anchor="2026-01-10",
        tier4_row={}, quote_rows=())
    refusal = nsb._project_member_features(
        event.key, event, _binding(), ("signal",), as_of="2026-01-10")

    assert isinstance(refusal, NativeScoreBatchRowRefusal)
    assert (refusal.code, refusal.detail) == (
        "MISSING_STAGED_INPUT",
        "required member feature is missing from staged rows")