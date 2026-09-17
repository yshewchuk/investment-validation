from __future__ import annotations

import pandas as pd

from engine.score import Phase4TraceCollector, ScoreRequest, ScoreResult, Scorer


def test_collector_is_opt_in_and_default_finish_does_not_mutate_result() -> None:
    scorer = Scorer.__new__(Scorer)
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )

    finished = scorer._finish_phase4_trace(None, result)

    assert finished is result
    assert not hasattr(result, "_phase4_trace")


def test_collector_records_json_safe_stage_documents_and_refusal_identity() -> None:
    collector = Phase4TraceCollector()
    collector.begin(ScoreRequest(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
        event_date=pd.Timestamp("2026-01-08"),
    ))
    collector.record(
        "features",
        {"frame": pd.DataFrame({"x": [1.0], "missing": [float("nan")]})},
        {"as_of": pd.Timestamp("2026-01-02")},
    )
    result = ScoreResult(
        ticker="ABC",
        strategy="STR-THRU",
        as_of=pd.Timestamp("2026-01-02"),
    )
    result.flag("NO_CHAIN")
    collector.finish(result)

    document = collector.document()

    assert document["schema_version"] == "phase4_legacy_trace.v1.0"
    assert document["status"] == "refused"
    assert document["request"]["event_date"] == "2026-01-08"
    assert document["stages"]["features"]["input"]["frame"][0] == {
        "x": 1.0,
        "missing": None,
    }
    assert document["stages"]["serialization"]["status"] == "refused"
