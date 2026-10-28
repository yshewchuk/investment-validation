# packages: engine.v2.models
# land: always-run
from __future__ import annotations

import json
from pathlib import Path

import pytest

from checks import live_experiment_isolation as guard


def test_violations_are_direct_ast_imports_in_live_roots(tmp_path: Path):
    model = tmp_path / "engine/v2/models"
    model.mkdir(parents=True)
    (model / "live.py").write_text(
        "import experiments.runner as runner\n"
        "from engine.v2.foundation import experiment_holdouts as holdouts\n"
        "from engine.v2 import research as research_pkg\n"
        "from engine.v2.foundation.experiment_holdouts import Holdouts\n"
        "from unrelated import experiments_helper\n"
        "note = 'import experiments.hidden'\n"
        "from ..research import report\n"
        "from ....experiments import beyond_top\n"
        "def lazy():\n"
        "    import experiments.lazy\n"
    )
    ops = tmp_path / "engine/v2/ops"
    ops.mkdir(parents=True)
    (ops / "training.py").write_text("import experiments.training\n")
    (ops / "retrain_due.py").write_text("from engine.v2.research import due\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools/phase5_job.py").write_text(
        "from engine.v2.foundation import final_holdout\n"
    )
    (tmp_path / "tools/other.py").write_text("import experiments.ignored\n")

    found = guard.violations(tmp_path)

    assert [(item["path"], item["line"], item["target"]) for item in found] == [
        ("engine/v2/models/live.py", 1, "experiments.runner"),
        ("engine/v2/models/live.py", 2, "engine.v2.foundation.experiment_holdouts"),
        ("engine/v2/models/live.py", 3, "engine.v2.research"),
        ("engine/v2/models/live.py", 4, "engine.v2.foundation.experiment_holdouts"),
        ("engine/v2/models/live.py", 7, "engine.v2.research"),
        ("engine/v2/models/live.py", 10, "experiments.lazy"),
        ("engine/v2/ops/retrain_due.py", 1, "engine.v2.research"),
        ("engine/v2/ops/training.py", 1, "experiments.training"),
        ("tools/phase5_job.py", 1, "engine.v2.foundation.final_holdout"),
    ]


def test_live_files_fails_when_expected_root_is_missing(tmp_path: Path):
    with pytest.raises(ValueError, match="missing expected live scan root"):
        guard.live_files(tmp_path)
    for rel in ("engine/v2/models", "engine/v2/ops", "tools"):
        (tmp_path / rel).mkdir(parents=True)
    (tmp_path / "engine/v2/models/live.py").touch()
    with pytest.raises(ValueError, match="missing expected live scan files"):
        guard.live_files(tmp_path)


def test_compare_requires_exact_inventory_and_counts_duplicate_occurrences():
    item = {"path": "engine/v2/models/a.py", "line": 1, "target": "experiments.x"}
    shifted = {**item, "line": 2}
    assert guard.compare([item], [item]) == ([], [])
    assert guard.compare([item, item], [item]) == ([item], [])
    assert guard.compare([], [item]) == ([], [item])
    assert guard.compare([shifted], [item]) == ([shifted], [item])


def test_regular_run_fails_for_added_and_missing_inventory(tmp_path: Path, monkeypatch):
    item = {"path": "engine/v2/models/a.py", "line": 1, "target": "experiments.x"}
    extra = {**item, "line": 2}
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps([item]))
    monkeypatch.setattr(guard, "BASELINE_PATH", baseline)
    monkeypatch.setattr(guard, "violations", lambda: [item])
    assert guard.main([]) == 1
    monkeypatch.setattr(guard, "violations", lambda: [item, extra])
    assert guard.main([]) == 1
    baseline.write_text(json.dumps([item, extra]))
    monkeypatch.setattr(guard, "violations", lambda: [])
    assert guard.main([]) == 1


def test_write_baseline_allows_shrink_and_refuses_growth(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    original = [{"path": "engine/v2/models/a.py", "line": 1, "target": "experiments.x"}]
    monkeypatch.setattr(guard, "BASELINE_PATH", baseline)
    monkeypatch.setattr(guard, "violations", lambda: original)
    assert guard.main(["--write-baseline"]) == 1
    assert not baseline.exists()
    monkeypatch.setattr(guard, "violations", lambda: [])

    assert guard.main(["--write-baseline"]) == 0
    assert json.loads(baseline.read_text()) == []
    baseline.write_text(json.dumps(original))
    monkeypatch.setattr(guard, "violations", lambda: original + [original[0]])

    assert guard.main(["--write-baseline"]) == 1
    assert json.loads(baseline.read_text()) == original


def test_bounded_read_refuses_oversized_input(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_bytes(b"12345")
    with pytest.raises(ValueError, match="exceeds 4 bytes"):
        guard._read_bounded(source, 4)


def test_repository_guard_matches_committed_baseline():
    assert guard.main([]) == 0
