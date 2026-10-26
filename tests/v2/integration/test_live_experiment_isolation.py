from __future__ import annotations

# packages: ops

from pathlib import Path

import pytest

from checks import live_experiment_isolation as isolation


def _rules(findings):
    return {item.rule for item in findings}


def test_tracked_repository_passes_real_isolation_scan():
    assert isolation.scan_repository(Path(__file__).resolve().parents[3]) == []


def test_production_training_cannot_import_holdout_or_research():
    files = {
        "engine/v2/models/retrain_job.py":
            b"from engine.v2.foundation.experiment_holdouts import final_month\n"
            b"import engine.v2.research.population\n",
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 2


def test_experiments_cannot_import_live_release_or_ledger_modules():
    files = {
        "experiments/run.py":
            b"from engine.v2.models import deployment\n"
            b"import engine.v2.ledger.rows\n",
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 2


def test_experiments_cannot_name_live_roots_but_may_name_own_ledger():
    files = {
        "experiments/run.py": (
            b"MODEL_RELEASE_ROOT = 'x'\n"
            b"POINTER = 'deployment/DEPLOYED'\n"
            b"LEDGER = 'LEDGER.csv'\n"
        ),
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-live-root"}
    assert len(findings) == 2


def test_new_production_root_caller_requires_explicit_review():
    findings = isolation.check_files({
        "experiments/new_runner.py": b"production_release_root()\n",
    })
    assert _rules(findings) == {"unlisted-root-caller"}


def test_aliased_production_root_caller_is_still_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models.deployment import production_release_root as root\n"
            b"root()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}


@pytest.mark.parametrize("body", [b"def broken(:\n", b"\xff"])
def test_invalid_source_is_a_finding_not_a_clean_partial_scan(body):
    findings = isolation.check_files({"experiments/broken.py": body})
    assert _rules(findings) == {"source-invalid"}


def test_git_listing_failure_refuses_instead_of_skipping_scan(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(isolation.subprocess, "run", fail)
    with pytest.raises(isolation.IsolationCheckError, match="complete tracked-source scan refused"):
        isolation.scan_repository(tmp_path)
