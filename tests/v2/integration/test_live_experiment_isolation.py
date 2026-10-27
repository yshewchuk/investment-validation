from __future__ import annotations

import json
from pathlib import Path

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
    )
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
        ("tools/phase5_job.py", 1, "engine.v2.foundation.final_holdout"),
    ]


def test_compare_requires_exact_inventory_and_counts_duplicate_occurrences():
    item = {"path": "engine/v2/models/a.py", "line": 1, "target": "experiments.x"}
    assert guard.compare([item], [item]) == ([], [])
    assert guard.compare([item, item], [item]) == ([item], [])
    assert guard.compare([], [item]) == ([], [item])


def test_write_baseline_allows_shrink_and_refuses_growth(tmp_path: Path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    original = [{"path": "engine/v2/models/a.py", "line": 1, "target": "experiments.x"}]
    monkeypatch.setattr(guard, "BASELINE_PATH", baseline)
    monkeypatch.setattr(guard, "violations", lambda: [])

    assert guard.main(["--write-baseline"]) == 0
    assert json.loads(baseline.read_text()) == []
    baseline.write_text(json.dumps(original))
    monkeypatch.setattr(guard, "violations", lambda: original + [original[0]])

    assert guard.main(["--write-baseline"]) == 1
    assert json.loads(baseline.read_text()) == original
