import json
from dataclasses import make_dataclass
from types import SimpleNamespace

import pytest

import engine.v2.ops.native_score_batch as native_score_batch
import engine.v2.scoring.release_bindings as release_bindings
from engine.v2.models.admissible_table import N_ADMISSIBLE_TABLE_ID
from engine.v2.ops.native_score_batch import (
    NativeScoreBatchRowRefusal,
    _STRATEGY_REGISTRY,
    _chooser_bundle_or_refusal,
)
from engine.v2.scoring.release_bindings import ModelNotReady, select_chooser_state


_MENU = ("TWIN-P", "TWIN-P5", "CND-PS", "BFLY-P", "BFLY-P5", "RAMP7", "CTR5")


def _binding(*pools):
    return SimpleNamespace(
        chooser_analog_artifacts=tuple(pools),
        chooser_admissible_artifacts=(),
    )


def test_chooser_is_declared_only_for_menu_member_specs():
    assert all(_STRATEGY_REGISTRY[name].chooser for name in _MENU)
    assert not _STRATEGY_REGISTRY["STR-THRU"].chooser
    assert not _STRATEGY_REGISTRY["STR-RUNUP"].chooser


def test_chooser_selector_requires_exact_session_cutoff():
    binding = _binding(SimpleNamespace(cutoff="2026-10-26"))
    with pytest.raises(ModelNotReady):
        select_chooser_state(
            binding, event_date="2026-10-27", as_of="2026-10-27")


@pytest.mark.parametrize("bad_as_of", ["2026-10-27x", "2026-10-27garbage"])
def test_chooser_selector_rejects_malformed_as_of_with_populated_cache(bad_as_of):
    binding = _binding()
    cache = {(id(binding), "chooser:2026-10-01"): object()}
    with pytest.raises(ModelNotReady):
        select_chooser_state(
            binding, "2026-10-27", bad_as_of, fold_cache=cache)


def test_chooser_selector_refuses_fold_starting_on_decision_session_before_cache():
    binding = SimpleNamespace(
        chooser_analog_artifacts=(SimpleNamespace(cutoff="2026-10-01"),),
        chooser_admissible_artifacts=(
            SimpleNamespace(key=(N_ADMISSIBLE_TABLE_ID, "v1")),
        ),
    )
    cache = {(id(binding), "chooser:2026-10-01"): object()}
    with pytest.raises(ModelNotReady):
        select_chooser_state(
            binding, "2026-10-01", "2026-10-01", fold_cache=cache)


def test_missing_chooser_state_becomes_typed_member_refusal():
    binding = _binding()
    refusal = _chooser_bundle_or_refusal(
        object(), SimpleNamespace(strategy="TWIN-P"), binding=binding,
        event_date="2026-10-27", as_of="2026-10-27", fold_cache={})
    assert isinstance(refusal, NativeScoreBatchRowRefusal)
    assert refusal.code == "MODEL_NOT_READY"
    assert refusal.detail == "DYN-SV chooser state missing or unusable"


def test_chooser_selector_selects_folds_and_caches(monkeypatch):
    roles = {"implied_t1": "pred_im_t1_d14", "runup_move": "pred_runup_abs_move_d14"}
    catalog = {"members": [
        {"member_id": f"tier4_folds:{role}", "status": "STAGED",
         "objects": [{"serving_fold": {
             "policy": {"role": role}, "fold_start": "2026-09-01"}}]}
        for role in roles
    ]}
    binding = SimpleNamespace(
        chooser_analog_artifacts=(
            SimpleNamespace(cutoff="2026-10-26", pool_id="stale", content_hash="h-old"),
            SimpleNamespace(cutoff="2026-10-27", pool_id="analog", content_hash="h-a"),
        ),
        chooser_admissible_artifacts=(SimpleNamespace(
            key=(N_ADMISSIBLE_TABLE_ID, "v1"), table_id=N_ADMISSIBLE_TABLE_ID,
            version="v1", content_hash="h-t"),),
        chooser_fold_catalog_json=json.dumps(catalog),
        chooser_parent_manifest=object(), frozen_inference=object(),
    )
    monkeypatch.setattr(
        release_bindings, "select_gate_forecast_source",
        lambda binding, event_date, as_of: SimpleNamespace(
            pool_pred="size-pred", pool_res="size-res", interval_floor=0.1),
    )

    def fake_from_document(cls, doc):
        role = doc["policy"]["role"]
        return SimpleNamespace(
            policy=SimpleNamespace(role=role, interval_floor=0.2),
            fold_start=doc["fold_start"], producer_output=roles[role])

    monkeypatch.setattr(release_bindings, "from_document", fake_from_document)
    loader_calls = []

    def fake_load_chooser_serving_fold(
        inference, descriptor, *, verified_parent_manifest, verified_catalog
    ):
        role = descriptor.policy.role
        loader_calls.append(role)
        return SimpleNamespace(
            pool_pred=f"{role}-pred", pool_res=f"{role}-res",
            release=SimpleNamespace(
                bindings=(SimpleNamespace(binding_id=f"fold:{role}"),)))

    monkeypatch.setattr(
        release_bindings, "load_chooser_serving_fold", fake_load_chooser_serving_fold)
    cache = {}
    state = release_bindings.select_chooser_state(
        binding, "2026-09-27", "2026-10-27", fold_cache=cache)

    assert state.analog_pool.pool_id == "analog"
    assert state.admissible_table.table_id == N_ADMISSIBLE_TABLE_ID
    assert state.fold_pools["pred_abs_move"] == {
        "predictions": "size-pred", "residuals": "size-res", "interval_floor": 0.1}
    assert state.fold_pools["pred_im_t1_d14"] == {
        "predictions": "implied_t1-pred", "residuals": "implied_t1-res",
        "interval_floor": 0.2}
    assert state.fold_pools["pred_runup_abs_move_d14"] == {
        "predictions": "runup_move-pred", "residuals": "runup_move-res",
        "interval_floor": 0.2}
    assert state.producer_bindings["pred_im_t1_d14"].binding_id == "fold:implied_t1"
    assert state.producer_bindings["pred_runup_abs_move_d14"].binding_id == "fold:runup_move"
    assert loader_calls == ["implied_t1", "runup_move"]

    release_bindings.select_chooser_state(
        binding, "2026-09-27", "2026-10-27", fold_cache=cache)
    assert len(loader_calls) == 2


def test_chooser_bundle_wires_producer_bindings_and_pools(monkeypatch):
    implied = SimpleNamespace(binding_id="fold:implied_t1")
    runup = SimpleNamespace(binding_id="fold:runup_move")
    existing = SimpleNamespace(binding_id="main")
    state = SimpleNamespace(
        producer_bindings={
            "pred_im_t1_d14": implied,
            "pred_runup_abs_move_d14": runup,
        },
        analog_pool=SimpleNamespace(
            pool_id="analog", cutoff="2026-10-27", content_hash="h-a"),
        admissible_table=SimpleNamespace(
            table_id=N_ADMISSIBLE_TABLE_ID, version="v1", content_hash="h-t"),
        fold_pools={"fold": "pools"},
    )
    monkeypatch.setattr(
        native_score_batch, "select_chooser_state",
        lambda binding, event_date, as_of, *, fold_cache: state)

    Release = make_dataclass("Release", [("bindings", tuple)], frozen=True)
    Bundle = make_dataclass(
        "Bundle",
        [
            ("model_release", object),
            ("chooser_recipe", object),
            ("chooser_fold_pools", object),
            ("chooser_analog_pool", object),
            ("chooser_admissible_table", object),
        ],
        frozen=True,
    )
    bundle = Bundle(
        model_release=Release((existing,)),
        chooser_recipe=None,
        chooser_fold_pools=None,
        chooser_analog_pool=None,
        chooser_admissible_table=None,
    )
    binding = SimpleNamespace(
        model_identity={
            "chooser:DYN-SV": SimpleNamespace(binding_id="chooser:dynsv"),
        },
    )

    result = native_score_batch._apply_declared_chooser(
        bundle, object(), _STRATEGY_REGISTRY["TWIN-P"],
        SimpleNamespace(calendar_row={"event_date": "2026-10-27"}),
        binding=binding, as_of="2026-10-27", fold_cache={})

    assert result.chooser_recipe["binding_id"] == "chooser:dynsv"
    assert result.chooser_recipe["producers"]["pred_im_t1_d14"]["binding_id"] == "fold:implied_t1"
    assert result.chooser_recipe["producers"]["pred_runup_abs_move_d14"]["binding_id"] == "fold:runup_move"
    assert result.chooser_recipe["analog_pool"] == {
        "pool_id": "analog", "cutoff": "2026-10-27", "content_hash": "h-a"}
    assert result.chooser_recipe["admissible_table"] == {
        "table_id": N_ADMISSIBLE_TABLE_ID, "version": "v1", "content_hash": "h-t"}
    assert result.model_release.bindings == (existing, implied, runup)
    assert result.chooser_fold_pools == {"fold": "pools"}
    assert result.chooser_analog_pool.pool_id == "analog"
