"""The published NATIVE_INPUT_BUILD_FAILED detail: exception class and
fixed-vocabulary answer-field names only, never staged keys or values."""
from __future__ import annotations

from engine.v2.ops import native_score_batch
from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
from engine.v2.scoring.source_inputs import _reject_answers
from tests.v2.ops.test_v2_ops_native_score_batch import (
    _assemble,
    _event_inputs,
    _stage_release,
)


def test_answer_field_in_source_publishes_class_and_field_names_only(
        tmp_path, monkeypatch):
    # The real _reject_answers raises from a payload whose nested key and value
    # are staged data; only the real refusal formatting is under test.
    def _raise_real(bundle):
        _reject_answers("metadata", {"gate_pass": {"SECRETKEY": 987654}})

    monkeypatch.setattr(native_score_batch, "build_native_score_inputs", _raise_real)
    binding = _stage_release(tmp_path)
    assembled, refusals = _assemble(binding, [_event_inputs()])
    assert assembled == {}
    assert len(refusals) == 1
    refusal = refusals[0]
    assert isinstance(refusal, NativeScoreBatchRowRefusal)
    assert refusal.code == "NATIVE_INPUT_BUILD_FAILED"
    assert refusal.detail == "native input build failed: AnswerFieldsInSource ['gate_pass']"
    assert "SECRETKEY" not in refusal.detail
    assert "987654" not in refusal.detail


def test_plain_value_error_publishes_class_name_only(tmp_path, monkeypatch):
    def _raise_plain(bundle):
        raise ValueError("SECRETKEY=987654 staged shape")

    monkeypatch.setattr(native_score_batch, "build_native_score_inputs", _raise_plain)
    binding = _stage_release(tmp_path)
    assembled, refusals = _assemble(binding, [_event_inputs()])
    assert assembled == {}
    assert len(refusals) == 1
    assert refusals[0].code == "NATIVE_INPUT_BUILD_FAILED"
    assert refusals[0].detail == "native input build failed: ValueError"