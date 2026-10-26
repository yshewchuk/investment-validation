"""One staged row carrying a calculated answer field refuses only that row
(issue #630): the real assemble path, no monkeypatching."""
from __future__ import annotations

from engine.v2.ops.native_score_batch import NativeScoreBatchRowRefusal
from tests.v2.ops.test_v2_ops_native_score_batch import (
    _assemble,
    _calendar_row,
    _event_inputs,
    _stage_release,
)


def test_answer_named_feature_refuses_that_row_and_scores_the_other(tmp_path):
    binding = _stage_release(tmp_path)
    bad = _event_inputs(
        ticker="BAD", calendar_row=_calendar_row(ticker="BAD", event_id="evt-bad"))
    bad.panel_row["signal"] = {"gate_pass": 1}
    clean = _event_inputs(
        ticker="CLEAN",
        calendar_row=_calendar_row(ticker="CLEAN", event_id="evt-clean"))
    assembled, refusals = _assemble(binding, [bad, clean])
    assert len(refusals) == 1
    refusal = refusals[0]
    assert isinstance(refusal, NativeScoreBatchRowRefusal)
    assert refusal.key.ticker == "BAD"
    assert refusal.code == "NATIVE_INPUT_BUILD_FAILED"
    assert refusal.detail == "native input build failed: AnswerFieldsInSource ['gate_pass']"
    assert len(assembled) + len(refusals) == 2