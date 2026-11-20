"""Search parent memory P8: worker stage memory lines at a bounded cadence; logging only."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from experiments import size_search_binding as binding
from experiments import v2_magnitude_search_run as run_mod


@pytest.fixture
def fresh_log(monkeypatch):
    monkeypatch.setattr(run_mod, "_WORKER_LOG", {"slices": 0, "fits": 0})


def test_cadence_logs_first_three_and_every_twentieth(fresh_log):
    hits = [n for n in range(1, 41) if run_mod._worker_due("slices")]
    assert hits == [1, 2, 3, 20, 40]


def test_worker_trial_logs_worker_after_fit(fresh_log, monkeypatch, capsys):
    panel = {"panel": object()}
    monkeypatch.setattr(binding, "_PANEL_MEMO", panel)
    trials_obj = Mock()
    registration = SimpleNamespace(run_id="run")
    monkeypatch.setattr(run_mod, "_PARALLEL", (None, None, trials_obj, registration, {"folds": []}))
    monkeypatch.setattr(run_mod, "trial_key", lambda plan, *args: "key")
    monkeypatch.setattr(run_mod.os, "sched_setaffinity", lambda pid, cpus: None)
    assert run_mod._parallel_trial_task(((2024, 2023, "overall"), 0, {"memory_bytes": 1})) is None
    assert "worker after fit" in capsys.readouterr().err
    assert panel == {}


class FakeTrials:
    def __init__(self, fail=False):
        self._memo, self.fail, self.calls = {}, fail, 0

    def _cohort_inputs(self, cohort, inner, plan=None):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        if cohort not in self._memo:
            self._memo[cohort] = {"rows": [inner]}
        return self._memo[cohort]


def test_slice_wrapper_logs_only_a_new_cohort_and_returns_unchanged(fresh_log, capsys):
    trials = FakeTrials()
    run_mod._worker_stage(trials)
    first = trials._cohort_inputs("c1", 2024)
    assert "worker slice loaded" in capsys.readouterr().err
    capsys.readouterr()
    assert trials._cohort_inputs("c1", 2024) is first
    assert "worker slice loaded" not in capsys.readouterr().err
    assert trials.calls == 2


def test_slice_wrapper_propagates_original_exception(fresh_log):
    trials = FakeTrials(fail=True)
    run_mod._worker_stage(trials)
    with pytest.raises(RuntimeError, match="boom"):
        trials._cohort_inputs("c1", 2024)
