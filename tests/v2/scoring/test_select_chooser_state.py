"""Tests for select_chooser_state."""

from types import SimpleNamespace

import pytest

from engine.v2.models.admissible_table import N_ADMISSIBLE_TABLE_ID
from engine.v2.scoring.release_bindings import (
    _CHOOSER_POOL_MEMBER_ID,
    _CHOOSER_TABLE_MEMBER_ID,
    _chooser_fold_descriptor,
    ChooserState,
    ModelNotReady,
    select_chooser_state,
)


def _binding(*, analog_pool, table):
    return SimpleNamespace(
        chooser_analog_artifacts=analog_pool,
        chooser_admissible_artifacts=table,
        chooser_fold_references={},
        chooser_authoring_fold_contexts={},
        chooser_fold_catalog_json="{}",
        frozen_inference=SimpleNamespace(),
        release_id="test-release",
    )


def _artifact(cutoff, key="default"):
    return SimpleNamespace(cutoff=cutoff, key=key)


def _table(table_id, version):
    return SimpleNamespace(key=(table_id, version))


def _patch_fold_state(monkeypatch, result=None, error=None):
    def _fake_load(*args, **kwargs):
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(
        "engine.v2.scoring.release_bindings._load_chooser_fold_state",
        _fake_load,
    )


def test_select_chooser_state_success(monkeypatch):
    pool = _artifact("2026-10-02")
    table = _table(N_ADMISSIBLE_TABLE_ID, "v1")
    binding = _binding(analog_pool=[pool], table=[table])
    captured = {}

    def _fake_load(binding_arg, event_date_arg, as_of_arg, fold_start_arg, decision_cutoff_arg):
        captured["event_date"] = event_date_arg
        captured["as_of"] = as_of_arg
        captured["fold_start"] = fold_start_arg
        captured["decision_cutoff"] = decision_cutoff_arg
        return (
            {"pred_abs_move": {"predictions": (9.0, 10.0), "residuals": (11.0, 12.0),
                               "interval_floor": 0.9},
             "pred_im_t1_d14": {"predictions": (1.0, 2.0), "residuals": (3.0, 4.0),
                                "interval_floor": 0.5},
             "pred_runup_abs_move_d14": {"predictions": (5.0, 6.0), "residuals": (7.0, 8.0),
                                         "interval_floor": 0.7}},
            {"pred_im_t1_d14": "d1", "pred_runup_abs_move_d14": "d2"},
            {"pred_im_t1_d14": {"training_cutoff": "2026-08-31",
                                "fold_start": "2026-09-01",
                                "descriptor_hash": "h1"},
             "pred_runup_abs_move_d14": {"training_cutoff": "2026-08-31",
                                         "fold_start": "2026-09-01",
                                         "descriptor_hash": "h2"}},
            ("h1", "h2"),
        )

    monkeypatch.setattr(
        "engine.v2.scoring.release_bindings._load_chooser_fold_state",
        _fake_load,
    )

    result = select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert isinstance(result, ChooserState)
    assert result.analog_pool is pool
    assert result.admissible_table is table
    assert result.fold_pools["pred_im_t1_d14"]["predictions"] == (1.0, 2.0)
    assert result.fold_pools["pred_runup_abs_move_d14"]["residuals"] == (7.0, 8.0)
    assert result.fold_pools["pred_abs_move"]["interval_floor"] == 0.9
    assert result.producer_bindings == {"pred_im_t1_d14": "d1", "pred_runup_abs_move_d14": "d2"}
    assert result.fold_provenance["pred_im_t1_d14"] == {
        "training_cutoff": "2026-08-31",
        "fold_start": "2026-09-01",
        "descriptor_hash": "h1",
    }
    assert result.fold_provenance["pred_runup_abs_move_d14"] == {
        "training_cutoff": "2026-08-31",
        "fold_start": "2026-09-01",
        "descriptor_hash": "h2",
    }
    assert result.artifact_refs == ("h1", "h2")
    assert captured == {
        "event_date": "2026-09-30",
        "as_of": "2026-10-02",
        "fold_start": "2026-09-01",
        "decision_cutoff": "2026-10-02",
    }


@pytest.mark.parametrize(
    ("event_date", "as_of"),
    [("not-a-date", "2026-10-02"), ("2026-09-30", "not-a-date")],
)
def test_select_chooser_state_invalid_dates(event_date, as_of):
    binding = _binding(
        analog_pool=[_artifact("2026-10-02")],
        table=[_table(N_ADMISSIBLE_TABLE_ID, "v1")],
    )

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, event_date, as_of)

    assert error.value.member_id == "chooser state"
    assert error.value.detail == "chooser state is missing or unusable"


def test_select_chooser_state_no_analog():
    binding = _binding(analog_pool=[], table=[])

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert error.value.member_id == _CHOOSER_POOL_MEMBER_ID
    assert error.value.detail == "no unique analog pool for cutoff"


def test_select_chooser_state_two_analogs():
    binding = _binding(
        analog_pool=[_artifact("2026-10-02"), _artifact("2026-10-02")],
        table=[],
    )

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert error.value.member_id == _CHOOSER_POOL_MEMBER_ID
    assert error.value.detail == "no unique analog pool for cutoff"


def test_select_chooser_state_no_table():
    binding = _binding(analog_pool=[_artifact("2026-10-02")], table=[])

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert error.value.member_id == _CHOOSER_TABLE_MEMBER_ID
    assert error.value.detail == "no unique table for key"


def test_select_chooser_state_two_tables():
    binding = _binding(
        analog_pool=[_artifact("2026-10-02")],
        table=[
            _table(N_ADMISSIBLE_TABLE_ID, "v1"),
            _table(N_ADMISSIBLE_TABLE_ID, "v1"),
        ],
    )

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert error.value.member_id == _CHOOSER_TABLE_MEMBER_ID
    assert error.value.detail == "no unique table for key"


def test_select_chooser_state_nested_not_ready(monkeypatch):
    binding = _binding(
        analog_pool=[_artifact("2026-10-02")],
        table=[_table(N_ADMISSIBLE_TABLE_ID, "v1")],
    )
    _patch_fold_state(monkeypatch, error=ModelNotReady("nested", "nested failure"))

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert error.value.member_id == "nested"
    assert error.value.detail == "nested failure"


def test_select_chooser_state_malformed_fold_error(monkeypatch):
    binding = _binding(
        analog_pool=[_artifact("2026-10-02")],
        table=[_table(N_ADMISSIBLE_TABLE_ID, "v1")],
    )
    _patch_fold_state(monkeypatch, error=ValueError("malformed"))

    with pytest.raises(ModelNotReady) as error:
        select_chooser_state(binding, "2026-09-30", "2026-10-02")

    assert error.value.member_id == "chooser state"
    assert error.value.detail == "chooser state is missing or unusable"


def test_chooser_fold_descriptor_parse_refusal(monkeypatch):
    def _refuse(*args, **kwargs):
        raise ValueError("/private/path")

    monkeypatch.setattr(
        "engine.v2.scoring.release_bindings.from_document", _refuse)
    catalog = {"members": [{
        "member_id": "tier4_folds:implied_t1",
        "status": "STAGED",
        "objects": [{"serving_fold": {"role": "implied_t1"}}],
    }]}

    with pytest.raises(ModelNotReady) as error:
        _chooser_fold_descriptor(
            SimpleNamespace(), catalog, "implied_t1", "2026-09-01", "2026-10-02")

    assert error.value.member_id == "tier4_folds:implied_t1"
    assert error.value.detail == "chooser fold descriptor is missing or unusable"
    assert error.value.__cause__ is None
