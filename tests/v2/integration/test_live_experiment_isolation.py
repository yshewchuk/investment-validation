from __future__ import annotations

# packages: ops
# land: always-run

import subprocess
from pathlib import Path

import pytest

from checks import live_experiment_isolation as isolation


def _rules(findings):
    return {item.rule for item in findings}


def test_tracked_repository_passes_real_isolation_scan(monkeypatch):
    original = isolation.check_files
    observed = {}

    def check_actual_sources(files):
        observed.update(files)
        return original(files)

    monkeypatch.setattr(isolation, "check_files", check_actual_sources)
    assert isolation.scan_repository(Path(__file__).resolve().parents[3]) == []
    assert "checks/live_experiment_isolation.py" in observed
    assert "engine/v2/models/deployment.py" in observed
    assert "experiments/lib.py" in observed


def test_production_training_cannot_import_holdout_or_research():
    files = {
        "engine/v2/models/retrain_job.py":
            b"from engine.v2.foundation.experiment_holdouts import final_month\n"
            b"from engine.v2.foundation.final_holdout import request_from_document\n"
            b"import engine.v2.research.population\n",
        "engine/v2/ops/training.py": b"import engine.v2.research.population\n",
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 4


def test_experiments_cannot_import_live_release_or_ledger_modules():
    files = {
        "experiments/run.py":
            b"from engine.v2.models import deployment\n"
            b"import engine.v2.ledger.rows\n",
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 2


def test_experiments_cannot_import_each_live_release_symbol():
    files = {
        "experiments/live.py": (
            b"from engine.v2.models import current_release\n"
            b"from engine.v2.models import current_pointer\n"
            b"from engine.v2.models import resolve_release\n"
            b"import engine.ledger\n"
            b"from engine.paths import LEDGER\n"
        ),
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 5


@pytest.mark.parametrize(
    "body",
    [
        b"import engine.v2.models as m\nm.current_release()\n",
        b"from engine.v2.models import *\n",
        b"from engine.v2 import models as m\nm.current_release()\n",
    ],
)
def test_experiments_cannot_reach_live_release_module_by_any_import_form(body):
    findings = isolation.check_files({"experiments/imports.py": body})
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_experiments_cannot_alias_the_live_ledger_path():
    findings = isolation.check_files({
        "experiments/ledger_path.py":
            b"import engine.paths as paths\nLEDGER_PATH = paths.LEDGER\n",
    })
    assert _rules(findings) == {"forbidden-live-root"}
    assert len(findings) == 1


def test_experiments_cannot_name_live_roots_but_may_name_own_ledger():
    files = {
        "experiments/run.py": (
            b"MODEL_RELEASE_ROOT = 'x'\n"
            b"POINTER = 'deployment/DEPLOYED'\n"
            b"ROOT.MODEL_RELEASE_ROOT = 'x'\n"
            b"LEDGER = 'LEDGER.csv'\n"
        ),
    }
    findings = isolation.check_files(files)
    assert _rules(findings) == {"forbidden-live-root"}
    assert len(findings) == 3


def test_new_production_root_caller_requires_explicit_review():
    findings = isolation.check_files({
        "experiments/new_runner.py": b"production_release_root()\n",
    })
    assert _rules(findings) == {"unlisted-root-caller"}


def test_aliased_production_root_caller_is_still_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root as root\n"
            b"root()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}


def test_relative_aliased_production_root_caller_is_still_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from ..models import production_release_root as root\n"
            b"root()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}


def test_attribute_aliased_production_root_caller_is_still_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import deployment\n"
            b"root = deployment.production_release_root\n"
            b"root()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}


@pytest.mark.parametrize("body", [b"def broken(:\n", b"\xff"])
def test_invalid_source_is_a_finding_not_a_clean_partial_scan(body):
    findings = isolation.check_files({"experiments/broken.py": body})
    assert _rules(findings) == {"source-invalid"}


def _listed_source(monkeypatch, name):
    monkeypatch.setattr(
        isolation, "_git_catalog", lambda root: (name + "\0").encode()
    )


def test_relative_research_import_into_models_release_is_blocked():
    findings = isolation.check_files({
        "engine/v2/research/sub/run.py": b"from ...models import deployment\n",
    })
    assert _rules(findings) == {"forbidden-import"}


def test_known_production_root_caller_is_allowed():
    assert isolation.check_files({
        "engine/v2/ops/training.py": b"production_deployment_root()\n",
    }) == []


def test_cli_main_returns_success_for_the_tracked_repository():
    assert isolation.main() == 0


def test_cli_main_returns_failure_when_scan_finds_a_violation(monkeypatch, capsys):
    finding = isolation.Finding("experiments/run.py", 1, "forbidden-import", "blocked")
    monkeypatch.setattr(isolation, "scan_repository", lambda: [finding])
    assert isolation.main() == 1
    assert "forbidden-import" in capsys.readouterr().err


def test_tracked_symlink_source_refuses_without_following_it(monkeypatch, tmp_path):
    (tmp_path / "target.py").write_bytes(b"pass\n")
    (tmp_path / "source.py").symlink_to(tmp_path / "target.py")
    _listed_source(monkeypatch, "source.py")
    with pytest.raises(isolation.IsolationCheckError, match="tracked source read refused"):
        isolation.scan_repository(tmp_path)


def test_oversized_tracked_source_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "MAX_SOURCE_BYTES", 4)
    (tmp_path / "source.py").write_bytes(b"12345")
    _listed_source(monkeypatch, "source.py")
    with pytest.raises(isolation.IsolationCheckError, match="exceeds 4 bytes"):
        isolation.scan_repository(tmp_path)


def test_aggregate_source_size_refuses(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "MAX_TOTAL_SOURCE_BYTES", 4)
    (tmp_path / "a.py").write_bytes(b"abc")
    (tmp_path / "b.py").write_bytes(b"de")
    monkeypatch.setattr(isolation, "_git_catalog", lambda root: b"a.py\0b.py\0")
    with pytest.raises(
        isolation.IsolationCheckError,
        match="aggregate Python source exceeds 4 bytes",
    ):
        isolation.scan_repository(tmp_path)


def test_git_listing_timeout_refuses_instead_of_skipping_scan(monkeypatch, tmp_path):
    def timeout(root):
        raise subprocess.TimeoutExpired("git", isolation.GIT_TIMEOUT_SECONDS)

    monkeypatch.setattr(isolation, "_git_catalog", timeout)
    with pytest.raises(isolation.IsolationCheckError, match="complete tracked-source scan refused"):
        isolation.scan_repository(tmp_path)


def test_git_listing_failure_refuses_instead_of_skipping_scan(monkeypatch, tmp_path):
    def fail(root):
        raise FileNotFoundError("git unavailable")

    monkeypatch.setattr(isolation, "_git_catalog", fail)
    with pytest.raises(isolation.IsolationCheckError, match="complete tracked-source scan refused"):
        isolation.scan_repository(tmp_path)


def test_catalog_overflow_refuses_instead_of_reading_unbounded_listing(monkeypatch):
    monkeypatch.setattr(isolation, "MAX_CATALOG_BYTES", 1)
    with pytest.raises(isolation.IsolationCheckError, match="exceeds 1 bytes"):
        isolation._git_catalog(isolation.ROOT)
