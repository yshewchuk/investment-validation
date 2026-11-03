"""Tests for select_chooser_state."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from engine.v2.foundation import content_hash
from engine.v2.models.admissible_table import N_ADMISSIBLE_TABLE_ID
from engine.v2.scoring import release_bindings
from engine.v2.scoring.release_bindings import (
    ChooserState,
    ModelNotReady,
    select_chooser_state,
)

_EVENT_DATE = "2026-09-30"
_AS_OF = "2026-10-02"
#: The earlier of the event/as-of months, derived by the real fold-start rule.
_FOLD_START = "2026-09-01"
#: The as-of month's fold start: the wrong-month decoy, never selected.
_WRONG_MONTH = "2026-10-01"
_TRAINING_CUTOFF = "2026-08-31"


@dataclass(frozen=True)
class _Policy:
    role: str
    interval_floor: float


def _binding(*, analog_pool, table, catalog_json="{}", references=None, contexts=None):
    return SimpleNamespace(
        chooser_analog_artifacts=analog_pool,
        chooser_admissible_artifacts=table,
        chooser_fold_references={} if references is None else references,
        chooser_authoring_fold_contexts={} if contexts is None else contexts,
        chooser_fold_catalog_json=catalog_json,
        frozen_inference=SimpleNamespace(),
        release_id="test-release",
    )


def _service_binding(**overrides):
    pool = _artifact(_AS_OF)
    table = _table(N_ADMISSIBLE_TABLE_ID, "v1")
    return _binding(analog_pool=[pool], table=[table], **overrides), pool, table


def _artifact(cutoff, key="default"):
    return SimpleNamespace(cutoff=cutoff, key=key)


def _table(table_id, version):
    return SimpleNamespace(key=(table_id, version))


def _serving_fold_doc(role, *, descriptor_hash, output, interval_floor,
                       fold_start=_FOLD_START, training_cutoff=_TRAINING_CUTOFF,
                       parent_release_id="authoring-release"):
    return {
        "role": role,
        "fold_start": fold_start,
        "training_cutoff": training_cutoff,
        "descriptor_hash": descriptor_hash,
        "producer_output": output,
        "parent_release_id": parent_release_id,
        "interval_floor": interval_floor,
    }


def _descriptor_from_document(_cls, doc):
    return SimpleNamespace(
        policy=_Policy(role=doc["role"], interval_floor=doc["interval_floor"]),
        fold_start=doc["fold_start"],
        training_cutoff=doc["training_cutoff"],
        descriptor_hash=doc["descriptor_hash"],
        producer_output=doc["producer_output"],
        parent_release_id=doc["parent_release_id"],
    )


def _fold_catalog(docs_by_role):
    return json.dumps({
        "members": [
            {"member_id": f"tier4_folds:{role}", "status": "STAGED",
             "objects": [{"serving_fold": doc} for doc in docs]}
            for role, docs in docs_by_role.items()
        ],
    })


def _size_source():
    return SimpleNamespace(
        pool_pred=(9.0, 10.0), pool_res=(11.0, 12.0),
        interval_floor=0.9, artifact_hash="sha256:size-artifact",
    )


def _reference(descriptor_hash, *, output):
    return SimpleNamespace(
        descriptor_hash=descriptor_hash,
        release_id="authoring-release",
        carrying_release_id="test-release",
        training_cutoff=_TRAINING_CUTOFF,
        fold_start=_FOLD_START,
    )


def test_select_chooser_state_assembles_derived_fold_state(monkeypatch):
    docs = {
        "implied_t1": [
            _serving_fold_doc("implied_t1", descriptor_hash="h-im",
                              output="pred_im_t1_d14", interval_floor=0.5),
            _serving_fold_doc("implied_t1", descriptor_hash="h-im-decoy",
                              output="pred_im_t1_d14", interval_floor=0.5,
                              fold_start=_WRONG_MONTH),
        ],
        "runup_move": [
            _serving_fold_doc("runup_move", descriptor_hash="h-ru",
                              output="pred_runup_abs_move_d14", interval_floor=0.7),
            _serving_fold_doc("runup_move", descriptor_hash="h-ru-decoy",
                              output="pred_runup_abs_move_d14", interval_floor=0.7,
                              fold_start=_WRONG_MONTH),
        ],
    }
    references = {
        ("pred_im_t1_d14", _FOLD_START): _reference("h-im", output="pred_im_t1_d14"),
        ("pred_runup_abs_move_d14", _FOLD_START): _reference(
            "h-ru", output="pred_runup_abs_move_d14"),
    }
    manifest = SimpleNamespace(release=SimpleNamespace(release_id="authoring-release"))
    contexts = {
        "h-im": (manifest, {"members": []}),
        "h-ru": (manifest, {"members": []}),
    }
    binding, pool, table = _service_binding(
        catalog_json=_fold_catalog(docs), references=references, contexts=contexts)

    source = _size_source()
    monkeypatch.setattr(
        release_bindings, "select_gate_forecast_source", lambda *_args, **_kwargs: source)
    monkeypatch.setattr(release_bindings, "from_document", _descriptor_from_document)
    loaded = []

    def _fake_load(_inference, descriptor, *, verified_parent_manifest, verified_catalog):
        loaded.append((descriptor.fold_start, descriptor.policy.role))
        return SimpleNamespace(
            pool_pred=(f"pred-{descriptor.policy.role}",),
            pool_res=(f"res-{descriptor.policy.role}",),
            release=SimpleNamespace(bindings=[f"binding-{descriptor.policy.role}"]),
        )

    monkeypatch.setattr(release_bindings, "load_chooser_serving_fold", _fake_load)

    result = select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert isinstance(result, ChooserState)
    assert result.analog_pool is pool
    assert result.admissible_table is table
    assert loaded == [(_FOLD_START, "implied_t1"), (_FOLD_START, "runup_move")]
    assert result.fold_pools["pred_abs_move"] == {
        "predictions": (9.0, 10.0), "residuals": (11.0, 12.0), "interval_floor": 0.9}
    assert result.fold_pools["pred_im_t1_d14"] == {
        "predictions": ("pred-implied_t1",), "residuals": ("res-implied_t1",),
        "interval_floor": 0.5}
    assert result.fold_pools["pred_runup_abs_move_d14"] == {
        "predictions": ("pred-runup_move",), "residuals": ("res-runup_move",),
        "interval_floor": 0.7}
    assert result.producer_bindings == {
        "pred_im_t1_d14": "binding-implied_t1",
        "pred_runup_abs_move_d14": "binding-runup_move",
    }
    assert result.fold_provenance["pred_im_t1_d14"] == {
        "training_cutoff": _TRAINING_CUTOFF,
        "fold_start": _FOLD_START,
        "descriptor_hash": "h-im",
    }
    assert result.fold_provenance["pred_runup_abs_move_d14"] == {
        "training_cutoff": _TRAINING_CUTOFF,
        "fold_start": _FOLD_START,
        "descriptor_hash": "h-ru",
    }
    implied_policy = _Policy(role="implied_t1", interval_floor=0.5)
    runup_policy = _Policy(role="runup_move", interval_floor=0.7)
    assert result.artifact_refs == (
        "sha256:size-artifact",
        content_hash({"interval_floor": 0.9}),
        "h-im",
        content_hash(release_bindings.to_document(implied_policy)),
        "h-ru",
        content_hash(release_bindings.to_document(runup_policy)),
    )


@pytest.mark.parametrize(
    ("event_date", "as_of"),
    [("not-a-date", "2026-10-02"), ("2026-09-30", "not-a-date")],
)
def test_select_chooser_state_invalid_dates(event_date, as_of):
    binding, _pool, _table = _service_binding()

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, event_date, as_of)

    assert error.value.member_id == "chooser state"
    assert error.value.detail == "chooser state is missing or unusable"


def test_select_chooser_state_no_analog():
    binding = _binding(analog_pool=[], table=[])

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "chooser_analog_pool"
    assert error.value.detail == "no unique analog pool for cutoff"


def test_select_chooser_state_two_analogs():
    binding = _binding(
        analog_pool=[_artifact(_AS_OF), _artifact(_AS_OF)],
        table=[],
    )

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "chooser_analog_pool"
    assert error.value.detail == "no unique analog pool for cutoff"


def test_select_chooser_state_no_table():
    binding = _binding(analog_pool=[_artifact(_AS_OF)], table=[])

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "admissible_table:dyn_sv"
    assert error.value.detail == "no unique table for key"


def test_select_chooser_state_two_tables():
    binding = _binding(
        analog_pool=[_artifact(_AS_OF)],
        table=[
            _table(N_ADMISSIBLE_TABLE_ID, "v1"),
            _table(N_ADMISSIBLE_TABLE_ID, "v1"),
        ],
    )

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "admissible_table:dyn_sv"
    assert error.value.detail == "no unique table for key"


def test_select_chooser_state_gate_forecast_source_refusal_propagates(monkeypatch):
    binding, _pool, _table = _service_binding()
    refusal = ModelNotReady("tier4_folds:size", "no staged size fold for 2026-09-01")

    def _refuse(*_args, **_kwargs):
        raise refusal

    monkeypatch.setattr(release_bindings, "select_gate_forecast_source", _refuse)

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value is refusal
    assert error.value.member_id == "tier4_folds:size"
    assert error.value.detail == "no staged size fold for 2026-09-01"


def test_select_chooser_state_malformed_catalog_is_generic_refusal(monkeypatch):
    binding, _pool, _table = _service_binding(catalog_json="{not json")
    monkeypatch.setattr(
        release_bindings, "select_gate_forecast_source",
        lambda *_args, **_kwargs: _size_source())

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "chooser state"
    assert error.value.detail == "chooser state is missing or unusable"


def test_select_chooser_state_missing_reference_refuses(monkeypatch):
    doc = _serving_fold_doc("implied_t1", descriptor_hash="h-im",
                            output="pred_im_t1_d14", interval_floor=0.5)
    binding, _pool, _table = _service_binding(catalog_json=_fold_catalog(
        {"implied_t1": [doc]}))
    monkeypatch.setattr(
        release_bindings, "select_gate_forecast_source",
        lambda *_args, **_kwargs: _size_source())
    monkeypatch.setattr(release_bindings, "from_document", _descriptor_from_document)

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "tier4_folds:implied_t1"
    assert error.value.detail == "reference is missing"


def test_select_chooser_state_noncausal_cutoff_refuses(monkeypatch):
    doc = _serving_fold_doc("implied_t1", descriptor_hash="h-im",
                            output="pred_im_t1_d14", interval_floor=0.5,
                            training_cutoff="2026-09-15")
    binding, _pool, _table = _service_binding(catalog_json=_fold_catalog(
        {"implied_t1": [doc]}))
    monkeypatch.setattr(
        release_bindings, "select_gate_forecast_source",
        lambda *_args, **_kwargs: _size_source())
    monkeypatch.setattr(release_bindings, "from_document", _descriptor_from_document)

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "tier4_folds:implied_t1"
    assert error.value.detail == "chooser fold descriptor is missing or unusable"


def test_select_chooser_state_descriptor_refusal_is_path_free(monkeypatch):
    def _refuse(_cls, _doc):
        raise ValueError("/private/path")

    binding, _pool, _table = _service_binding(catalog_json=json.dumps({
        "members": [{
            "member_id": "tier4_folds:implied_t1",
            "status": "STAGED",
            "objects": [{"serving_fold": {"role": "implied_t1"}}],
        }],
    }))
    monkeypatch.setattr(
        release_bindings, "select_gate_forecast_source",
        lambda *_args, **_kwargs: _size_source())
    monkeypatch.setattr(release_bindings, "from_document", _refuse)

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, _EVENT_DATE, _AS_OF)

    assert error.value.member_id == "tier4_folds:implied_t1"
    assert error.value.detail == "chooser fold descriptor is missing or unusable"
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True
    assert "/private/path" not in str(error.value)