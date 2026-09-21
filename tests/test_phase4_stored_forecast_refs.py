"""The stored crush forecast travels as an ADDRESS, and native resolves it.

Legacy ``Scorer._crush_forecast`` reads ``pred_iv_crush_30`` straight out of
the stored Tier-4 table for any event that has already printed, which is
every HISTORICAL row of the seven planned-exit DYN-SV strategies. That value
is an output of the system Phase 4 is testing, so capturing it and handing it
back to native makes the comparison circular: the check would pass with
native's whole retrieval path broken.

These tests pin the replacement. The capture emits the cell's address plus a
one-way content hash of the row and value together;
``checks/phase4_stored_forecasts.py`` opens the table, verifies the vintage,
reads the cell and only then checks what it read. The negative test
(:func:`test_supplying_the_value_through_the_reference_is_refused`) is the
load-bearing one: the new container must not become the hole in
``tools/phase4_release_assembler.py::_reject_answers``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import checks.phase4_stored_forecasts as stored_forecasts
from checks.phase4_stored_forecasts import (
    StoredForecastError,
    resolve_stored_forecasts,
    with_stored_forecasts,
)
from engine.v2.scoring.source_inputs import (
    STORED_REF_FIELDS,
    STORED_ROW_FIELDS,
    SourceBundle,
    _stored_forecast_refs,
    stored_forecast_row_hash,
)
from engine.v2.scoring.stages import (
    STAGE_NAMES,
    NativeScoreInputs,
    StageReceipt,
    _execute_local_forecast,
)
from tools.phase4_release_assembler import (
    ReleaseAssemblyError,
    _reject_answers,
    _reject_stored_refs,
)

VALUE = -17.25
MODEL_ID = "iv-crush-hgbr-v1"
FOLD_START = "2024-01-01"
EVENT = "2024-02-02"


def _table(tmp_path: Path) -> Path:
    """A stand-in Tier-4 forecasts table: one resolvable row, one NaN row."""
    path = tmp_path / "tier4_forecasts.parquet"
    pd.DataFrame({
        "ticker": ["AAA", "BBB"],
        "event_date": pd.to_datetime([EVENT, EVENT]).astype("datetime64[us]"),
        "pred_iv_crush_30": [VALUE, float("nan")],
        "pred_iv_crush_30_model_id": pd.array([MODEL_ID, MODEL_ID], dtype="string"),
        "pred_iv_crush_30_fold_start": pd.to_datetime(
            [FOLD_START, FOLD_START]).astype("datetime64[us]"),
    }).to_parquet(path)
    return path


def _ref(path: Path, *, value: float = VALUE, **overrides) -> dict:
    row = {
        "table": "tier4_forecasts",
        "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "column": "pred_iv_crush_30",
        "ticker": "AAA",
        "event_date": EVENT,
        "model_id": MODEL_ID,
        "fold_start": FOLD_START,
    }
    row.update(overrides)
    row = {key: row[key] for key in sorted(row)}
    return {"row": row, "row_hash": stored_forecast_row_hash(row, value)}


def _bundle(refs) -> SourceBundle:
    return SourceBundle(
        source_ref="stored-ref-test", strategy="STR-THRU", context={},
        raw_quotes={}, feature_vector={}, feature_missing_mask={},
        model_identity={}, forecast_recipes={}, model_artifact_refs={},
        residual_recipe={}, analog_recipe={}, gate_recipe={},
        stored_forecast_refs=refs)


def _inputs(ref, *, forecast_extra=None) -> NativeScoreInputs:
    forecast = {"stored_refs": {"pred_iv_crush_30": ref}}
    forecast.update(forecast_extra or {})
    return NativeScoreInputs(
        context={}, features={}, forecast=forecast, analogs={}, simulation={},
        gate={}, chooser={}, diagnostics={}, geometry=None, pricing=None,
        source_ref="stored-ref-test",
        stage_receipts=tuple(
            StageReceipt(stage=name, input_hash="in", output_hash="out", owner="test")
            for name in STAGE_NAMES if name not in ("diagnostics", "model")
        ),
    )


@pytest.fixture
def table(tmp_path, monkeypatch):
    path = _table(tmp_path)
    monkeypatch.setattr(
        stored_forecasts, "STORED_TABLES", {"tier4_forecasts": lambda: path})
    return path


# ---------------------------------------------------------------------------
# what travels
# ---------------------------------------------------------------------------

def test_the_reference_carries_no_value(table):
    ref = _ref(table)
    assert set(ref) == STORED_REF_FIELDS
    assert set(ref["row"]) == STORED_ROW_FIELDS
    serialized = json.dumps(ref)
    assert "value" not in serialized
    assert str(VALUE) not in serialized
    assert f"{VALUE:.2f}" not in serialized


def test_native_resolves_the_cell_from_the_table(table):
    """The DEFAULT reader, against a real parquet: native does the lookup."""
    inputs = _inputs(_ref(table))
    resolved = resolve_stored_forecasts(inputs)
    out = with_stored_forecasts(inputs, resolved)
    assert out.forecast["stored"]["pred_iv_crush_30"]["value"] == VALUE
    # The address survives resolution, so the scored inputs still record
    # where the value came from.
    assert out.forecast["stored_refs"]["pred_iv_crush_30"] == _ref(table)
    # The unresolved inputs are untouched.
    assert "stored" not in inputs.forecast


def test_resolve_is_a_noop_without_a_reference():
    inputs = _inputs({})
    bare = NativeScoreInputs(
        context={}, features={}, forecast={}, analogs={}, simulation={},
        gate={}, chooser={}, diagnostics={}, geometry=None, pricing=None,
        source_ref="bare", stage_receipts=inputs.stage_receipts)
    assert resolve_stored_forecasts(bare) is None
    assert with_stored_forecasts(bare, None) is bare


# ---------------------------------------------------------------------------
# the negative test: the reference must not become a loophole
# ---------------------------------------------------------------------------

def test_supplying_the_value_through_the_reference_is_refused(table):
    ref = _ref(table)

    # (a) beside the address.
    with pytest.raises(ReleaseAssemblyError) as err:
        _reject_stored_refs({"forecast": {"stored_refs": {
            "pred_iv_crush_30": {**ref, "value": VALUE}}}})
    assert "value" in str(err.value)

    # (b) hidden inside the address, where _ANSWER_FIELDS cannot see it
    # (nothing is named ``pred_iv_crush_30`` there).
    with pytest.raises(ReleaseAssemblyError) as err:
        _reject_stored_refs({"forecast": {"stored_refs": {
            "pred_iv_crush_30": {**ref, "row": {**ref["row"], "value": VALUE}}}}})
    assert "value" in str(err.value)

    # (c) as the RESOLVED block, which only the replay may produce. Both the
    # new structural rule and the original answer walk refuse it.
    resolved = {"forecast": {"stored": {"pred_iv_crush_30": {
        "value": VALUE, "row": ref["row"], "row_hash": ref["row_hash"]}}}}
    with pytest.raises(ReleaseAssemblyError, match="produced natively at replay"):
        _reject_stored_refs(resolved)
    with pytest.raises(ReleaseAssemblyError, match="calculated answer"):
        _reject_answers(resolved)

    # (d) the build side refuses it too, so a bundle cannot carry what a
    # trace may not.
    with pytest.raises(ValueError, match="unsupported recipe fields"):
        _stored_forecast_refs(_bundle(
            {"pred_iv_crush_30": {**ref, "value": VALUE}}))

    # And the honest reference still passes every one of those checks.
    _reject_stored_refs({"forecast": {"stored_refs": {"pred_iv_crush_30": ref}}})
    _reject_answers({"forecast": {"stored_refs": {"pred_iv_crush_30": ref}}})
    assert _stored_forecast_refs(
        _bundle({"pred_iv_crush_30": ref}))["pred_iv_crush_30"] == ref


def test_an_unresolved_reference_refuses_instead_of_falling_back(table):
    """Nobody resolved the address, and a competing recipe is standing right
    there. Taking it would compare a different forecast path with legacy's
    table read and call that parity."""
    inputs = _inputs(_ref(table), forecast_extra={
        "models": {"pred_iv_crush_30": {"intercept": -3.0, "coefficients": {}}}})
    output, invalid, flags = {}, set(), []
    declared = _execute_local_forecast(
        inputs, inputs.forecast, {}, output, invalid, flags, set())
    assert declared
    assert "UNRESOLVED_STORED_FORECAST:pred_iv_crush_30" in flags
    assert "pred_iv_crush_30" in invalid
    assert "pred_iv_crush_30" not in output


# ---------------------------------------------------------------------------
# every way resolution can fail, and the code it refuses with
# ---------------------------------------------------------------------------

def _reason(ref, **kwargs) -> str:
    with pytest.raises(StoredForecastError) as err:
        resolve_stored_forecasts(_inputs(ref), **kwargs)
    return err.value.reason


def test_resolution_refuses_a_table_of_another_vintage(table):
    assert _reason(_ref(table, table_sha256="0" * 64)) == (
        "STORED_FORECAST_TABLE_MISMATCH")


def test_resolution_refuses_a_key_the_table_does_not_hold(table):
    assert _reason(_ref(table, ticker="ZZZ")) == "STORED_FORECAST_ROW_MISSING"
    assert _reason(_ref(table, event_date="2024-03-03")) == (
        "STORED_FORECAST_ROW_MISSING")


def test_resolution_refuses_a_cell_the_table_holds_as_nan(table):
    # Legacy takes the stored branch only for a non-NaN cell.
    assert _reason(_ref(table, ticker="BBB")) == "STORED_FORECAST_VALUE_MISSING"


def test_resolution_refuses_a_row_written_by_another_model(table):
    def reader(path, ticker, event_date, column):
        return {"value": VALUE, "model_id": "someone-else", "fold_start": FOLD_START}

    assert _reason(_ref(table), reader=reader) == (
        "STORED_FORECAST_PROVENANCE_MISMATCH")

    def stale(path, ticker, event_date, column):
        return {"value": VALUE, "model_id": MODEL_ID, "fold_start": "2023-01-01"}

    assert _reason(_ref(table), reader=stale) == (
        "STORED_FORECAST_PROVENANCE_MISMATCH")


def test_resolution_refuses_a_cell_that_is_not_the_one_the_capture_hashed(table):
    def reader(path, ticker, event_date, column):
        return {"value": VALUE + 1.0, "model_id": MODEL_ID, "fold_start": FOLD_START}

    assert _reason(_ref(table), reader=reader) == (
        "STORED_FORECAST_ROW_HASH_MISMATCH")
    assert _reason({**_ref(table), "row_hash": "sha256:" + "0" * 64}) == (
        "STORED_FORECAST_ROW_HASH_MISMATCH")


def test_resolution_refuses_a_table_it_does_not_know_or_cannot_find(table, tmp_path):
    assert _reason(_ref(table, table="somewhere_else")) == (
        "STORED_FORECAST_TABLE_UNKNOWN")
    missing = tmp_path / "absent.parquet"
    stored_forecasts.STORED_TABLES = {"tier4_forecasts": lambda: missing}
    assert _reason(_ref(table)) == "STORED_FORECAST_TABLE_MISSING"


def test_resolution_refuses_a_malformed_reference(table):
    ref = _ref(table)
    assert _reason({"row": ref["row"]}) == "STORED_FORECAST_REF_MALFORMED"
    assert _reason({**ref, "row": {k: v for k, v in ref["row"].items()
                                   if k != "model_id"}}) == (
        "STORED_FORECAST_REF_MALFORMED")
    assert _reason(_ref(table, column="pred_iv_crush")) == (
        "STORED_FORECAST_REF_MALFORMED")
