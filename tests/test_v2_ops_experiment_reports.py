"""Runtime reachability evidence for the Phase 6 research capability rows.

``tests/test_v2_ops_experiment_job.py`` exercises only the synthetic fixture,
which writes the ``by engine.report v`` marker itself and never calls
``engine.report``. This module drives the real writer instead: a primary-mode
``run_experiment`` whose runner is the already-declared ``invoke_evaluate``
adapter (``engine.evaluate.evaluate`` -> ``engine.report``) against an
entirely in-repo, literal trade fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from engine import paths
from engine.v2.ops import experiments
from engine.v2.ops.experiments import experiment_spec_from_document

REPO = Path(__file__).resolve().parents[1]
EVALUATE_SPEC = {
    "id": "EXP-SPECB-REPORT-PROBE",
    "title": "real report reachability probe",
    "hypothesis": "REPORT.md is written by engine.report itself",
    "primary_spec": {"probe": True},
    "walk_forward": {"min_train_years": 1},
    "preregistered_at": "2020-01-01T00:00:00+00:00",
}


def _experiment_document():
    return {"experiment_id": "EXP-SPECB-REPORT", "hypothesis": "report plumbing",
            "primary_arm_id": "fixture", "arms": ["fixture"], "seed": 7,
            "folds": ["fold-1"], "economic_params": {"fill": "mid"},
            "price_source": "synthetic", "runner": "synthetic"}


def _trades():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.02, 0.1, 120)
    dates = pd.date_range("2020-01-01", periods=len(rets), freq="10D")
    frames = []
    for alpha in (0.0, 0.5, 1.0):
        frames.append(pd.DataFrame({
            "event_id": [f"E{i}" for i in range(len(rets))],
            "ticker": "T", "event_date": dates,
            "entry_date": dates - pd.Timedelta(days=1),
            "exit_date": dates + pd.Timedelta(days=1),
            "fill_alpha": float(alpha), "entry_cost": 1.0,
            "exit_value": 1.0 + rets, "ret": rets,
        }))
    return pd.concat(frames, ignore_index=True)


def test_primary_run_produces_a_real_engine_report(tmp_path):
    spec = experiment_spec_from_document(_experiment_document())
    assert paths.ROOT == REPO
    real_ledger = paths.ROOT / "experiments" / "LEDGER.csv"
    before = real_ledger.read_bytes() if real_ledger.is_file() else None

    def real_report_runner(*, run_dir, no_ledger):
        from engine.v2.ops.legacy_adapter import invoke_evaluate

        return invoke_evaluate(tmp_path, EVALUATE_SPEC, _trades(), run_dir=run_dir,
                               mc_paths=30, stress=False)

    receipt = experiments.run_experiment(spec, tmp_path, tmp_path,
                                         runner=real_report_runner, mode="primary")
    assert receipt["status"] == "succeeded"
    assert receipt["evidence"]["synthetic"] is False
    report_text = (tmp_path / "REPORT.md").read_text()
    assert "by engine.report v" in report_text
    assert "## 1. Headline (walk-forward OOS, worst/mid/best fills)" in report_text
    assert (tmp_path / "figures" / "alpha_breakeven.png").is_file()
    assert receipt["evidence"]["report_bytes"] > 0

    after = real_ledger.read_bytes() if real_ledger.is_file() else None
    assert after == before, "the real experiments/LEDGER.csv must never be touched"
