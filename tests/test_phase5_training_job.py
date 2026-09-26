"""Plan-only behavior of the P5-3 training job's frozen-state half.

``run_state_job`` used to build the full Tier-3 state and then throw it away
under ``--plan-only``; it must refuse before ``build_state`` is reached.
"""
import pytest

from tools import phase5_training_job as job


def test_run_state_job_plan_only_refuses_before_building(tmp_path, monkeypatch):
    calls = []

    def reached(*args, **kwargs):
        calls.append(args)
        raise AssertionError("build_state was reached")

    monkeypatch.setattr(job, "build_state", reached)
    out = tmp_path / "out"
    with pytest.raises(SystemExit):
        job.run_state_job("driver_residual_pool:size", out, plan_only=True)
    assert calls == []
    assert not out.exists()

    with pytest.raises(AssertionError, match="build_state was reached"):
        job.run_state_job("driver_residual_pool:size", out, plan_only=False)
    assert len(calls) == 1
