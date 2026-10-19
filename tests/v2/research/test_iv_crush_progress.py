"""Public planning refusals and real IV telemetry wiring with deterministic clocks."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from engine.v2.data.errors import DataError
from engine.v2.data.iv_crush_table import IV_CRUSH_CONTRACT
from engine.v2.research import event_history, progress
from engine.v2.research.iv_crush_inputs import preflight_iv_crush_targets
from tests.v2.research.magnitude_trial_support import history_event
from tests.v2.research.test_iv_crush_inputs import (
    MONTH,
    TABLE,
    committed,
    exact_problem,
    load,
    poison,
    target,
)


@pytest.mark.parametrize("candidates", [None, [], {}, pd.DataFrame(),
    pd.DataFrame({"ticker": ["SYNTH"]}),
    *[pd.DataFrame({"ticker": [ticker], "event_date": [day]}) for ticker, day in (
        (None, pd.Timestamp("2024-01-04")), (" ", pd.Timestamp("2024-01-04")),
        (" padded", pd.Timestamp("2024-01-04")), (1, pd.Timestamp("2024-01-04")),
        ("SYNTH", "bad"), ("SYNTH", None), ("SYNTH", "2024-01-04"),
        ("SYNTH", datetime(2024, 1, 4, 1)), ("SYNTH", datetime(2024, 1, 4, tzinfo=timezone.utc)))],
])
def test_public_planner_malformed_candidates_refuse_before_repository_access(candidates):
    with pytest.raises(DataError) as caught:
        event_history.plan_event_targets(None, None, candidates, IV_CRUSH_CONTRACT)
    exact_problem(caught, "CONTRACT_MISMATCH", "event target candidates are malformed")


def test_public_planner_downstream_errors_escape_candidate_translation(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(event_history, "_plans", boom)
    candidates = pd.DataFrame({"ticker": ["SYNTH"], "event_date": [pd.Timestamp("2024-01-04")]})
    with pytest.raises(RuntimeError, match="boom"):
        event_history.plan_event_targets(None, None, candidates, IV_CRUSH_CONTRACT)


class Clock:
    now = 0.

    def __call__(self):
        return self.now


def instrument(monkeypatch, repository):
    clock = Clock()
    original_progress, original_stage = progress.Progress, progress.stage
    def reporter(name, total, **kwargs):
        return original_progress(name, total, clock=clock, **kwargs)
    def stage(name, **kwargs):
        return original_stage(name, clock=clock, **kwargs)
    monkeypatch.setattr(progress, "_rss", lambda: ", peak RSS 12 MB")
    monkeypatch.setattr(progress, "Progress", reporter)
    monkeypatch.setattr(progress, "stage", stage)
    monkeypatch.setattr(event_history, "Progress", reporter)
    monkeypatch.setattr(event_history, "stage", stage)
    query, scan = event_history._target_query, repository.scan
    def planned(*args, **kwargs):
        result = query(*args, **kwargs)
        clock.now += 15
        return result
    def scanned(request, *, table_name, **kwargs):
        yield from scan(request, table_name=table_name, **kwargs)
        if table_name == TABLE:
            clock.now += 15
    monkeypatch.setattr(event_history, "_target_query", planned)
    monkeypatch.setattr(repository, "scan", scanned)


@pytest.mark.parametrize("mode", ["preflight", "load", "failure"])
def test_real_iv_stages_ticks_refusals_and_metadata_only_preflight(tmp_path, monkeypatch, capsys, mode):
    events = [history_event(day) for day in ("2024-01-04", "2024-03-05")]
    rows = [target(e, skipped=mode == "failure" and i == 1) for i, e in enumerate(events)]
    with committed(tmp_path, events, rows) as (_, repository, snapshot):
        instrument(monkeypatch, repository)
        capsys.readouterr()
        if mode == "preflight":
            poison(monkeypatch, repository, snapshot)
            result = preflight_iv_crush_targets(repository, snapshot, as_of_month=MONTH,
                                                event_ids=[e["event_id"] for e in events])
            assert result["event_ids"] == [e["event_id"] for e in events]
            assert all(name != TABLE for name, _ in repository.scans)
        elif mode == "failure":
            with pytest.raises(DataError) as caught:
                load(repository, snapshot)
            exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")
        else:
            assert load(repository, snapshot).crush_pct_iv30.tolist() == [-40., -40.]
        expected = [
            "  [IV target admission] start, peak RSS 12 MB",
            "  [plan event targets] start, peak RSS 12 MB",
            "  [plan history candidates] 2/2, 30s elapsed, 0.07/s, ETA 0s, peak RSS 12 MB",
            "  [plan event targets] done in 30s, peak RSS 12 MB",
            "  [IV target admission] done in 30s, peak RSS 12 MB",
        ]
        if mode != "preflight":
            expected.append("  [IV target reads] start, peak RSS 12 MB")
            if mode == "load":
                expected.append("  [IV targets] 2/2, 30s elapsed, 0.07/s, ETA 0s, peak RSS 12 MB")
            expected.append("  [IV target reads] " + ("done in" if mode == "load" else "failed after")
                            + " 30s, peak RSS 12 MB")
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err.splitlines() == expected


def test_broken_stderr_preserves_iv_result_and_original_refusal(tmp_path, monkeypatch):
    event = history_event("2024-05-06")
    attempts = []
    def broken(*args, **kwargs):
        attempts.append(args)
        raise OSError("closed stderr")
    with committed(tmp_path, [event], [target(event)]) as (_, repository, snapshot):
        monkeypatch.setattr("builtins.print", broken)
        assert load(repository, snapshot).crush_pct_iv30.tolist() == [-40.]
        with pytest.raises(DataError) as caught:
            load(repository, snapshot, labels_before="2024-05-08")
        exact_problem(caught, "EXPERIMENT_VARIANT_FAILED", "IV-crush label identity or availability is invalid")
        assert len(attempts) == 12
