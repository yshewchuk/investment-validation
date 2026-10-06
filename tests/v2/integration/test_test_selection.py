"""The package test selector.

# land: always-run
# packages: engine.v2.contracts, engine.v2.foundation, engine.v2.data, engine.v2.features, engine.v2.models, engine.v2.registry, engine.v2.domain.generation, engine.v2.domain.scenarios, engine.v2.domain.valuation, engine.v2.domain.simulation, engine.v2.scoring, engine.v2.evaluation, engine.v2.research, engine.v2.ledger, engine.v2.models.training, engine.v2.serving, engine.v2.ops, engine.v2.dashboard, engine.v2.diagnosis, engine.v2.parity

Path mapping, the reverse-import closure, integration declarations,
unmapped-path and unsafe-path failure, and a cache regression.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from checks import test_selection as ts  # noqa: E402


def _integration(root, name, body):
    directory = root / "tests/v2/integration"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body)
    return ts.INTEGRATION + name


# --------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------


def test_package_selection_is_the_reverse_closure_plus_meta(tmp_path):
    selection = ts.select(["engine/v2/data/store.py", "tests/v2/data/test_store.py"],
                          root=tmp_path)
    assert not selection.full_suite
    assert "tests/v2/data" in selection.targets
    assert "tests/v2/scoring" in selection.targets       # a higher layer may import data
    assert "tests/v2/contracts" not in selection.targets  # a lower layer may not
    assert set(ts.META_TESTS) <= set(selection.targets)


def test_longest_directory_selects_models_training_not_models(tmp_path):
    selection = ts.select(["engine/v2/models/training/fit.py"], root=tmp_path)
    assert not selection.full_suite
    assert "tests/v2/models/training" in selection.targets
    assert "tests/v2/models" not in selection.targets


@pytest.mark.parametrize("path", [
    "engine/v2/models/training.py", "tests/v2/models/training.py",
])
def test_conflicting_names_map_to_the_containing_package(tmp_path, path):
    selection = ts.select([path], root=tmp_path)
    assert not selection.full_suite
    assert "tests/v2/models" in selection.targets


def test_only_imports_and_sink_edges(tmp_path):
    foundation = ts.select(["engine/v2/foundation/env.py"], root=tmp_path)
    assert "tests/v2/parity" in foundation.targets          # parity only_imports 0.5
    assert "tests/v2/contracts" not in foundation.targets    # a lower layer may not
    dashboard = ts.select(["engine/v2/dashboard/app.py"], root=tmp_path)
    assert "tests/v2/diagnosis" in dashboard.targets         # a sink may import any package


@pytest.mark.parametrize("path", [
    "tools/mutation_pilot.py", "README.md", "tests/test_legacy.py", "checks/x.py",
    "engine/v2/__init__.py", ".github/workflows/x.yml",
])
def test_unmapped_paths_are_full_suite(tmp_path, path):
    selection = ts.select([path], root=tmp_path)
    assert selection.full_suite and path in selection.reason
    if path.startswith(".github/"):
        assert selection.reason == f"{path} is outside the layer map"


@pytest.mark.parametrize("path", [
    "/etc/passwd", "engine/v2/./ops/x.py", "engine/v2/ops/../../checks/x.py",
])
def test_traversal_paths_are_full_suite(tmp_path, path):
    selection = ts.select([path], root=tmp_path)
    assert selection.full_suite
    assert "unsafe" in selection.reason and path in selection.reason


def test_markdown_selects_meta_not_the_full_suite(tmp_path):
    selection = ts.select(["guides/anything.md", "engine/v2/models/ARCHITECTURE.md"],
                          root=tmp_path)
    assert not selection.full_suite
    assert set(ts.META_TESTS) <= set(selection.targets)


def test_integration_test_selects_itself_and_declared_packages(tmp_path):
    path = _integration(tmp_path, "test_x.py", "# packages: ops\n")
    selection = ts.select([path], root=tmp_path)
    assert path in selection.targets and "tests/v2/ops" in selection.targets
    assert path in ts.select(["engine/v2/ops/run.py"], root=tmp_path).targets


@pytest.mark.parametrize("body", [
    "", "# packages:\n", "# packages: ops, ops\n", "# packages: nope\n",
    "# packages: ops\n# packages: models\n",
])
def test_invalid_integration_declaration_is_full_suite(tmp_path, body):
    selection = ts.select([_integration(tmp_path, "test_bad.py", body)], root=tmp_path)
    assert selection.full_suite and selection.errors


def test_misplaced_declaration_on_a_package_test_is_full_suite(tmp_path):
    directory = tmp_path / "tests/v2/ops"
    directory.mkdir(parents=True)
    (directory / "test_x.py").write_text("# packages: ops\n")
    assert ts.select(["tests/v2/ops/test_x.py"], root=tmp_path).full_suite


def test_a_stale_cache_cannot_widen_or_replace_the_selection(tmp_path):
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache/test_selection.json").write_text(
        json.dumps({"targets": ["tests/"], "full_suite": True}))
    selection = ts.select(["engine/v2/dashboard/app.py"], root=tmp_path)
    assert not selection.full_suite and "tests/" not in selection.targets
    assert "tests/v2/dashboard" in selection.targets
