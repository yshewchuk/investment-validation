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


def test_annotated_assignment_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root\n"
            b"alias: object = production_release_root\n"
            b"alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [3]


def test_named_expression_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root\n"
            b"if (alias := production_release_root):\n"
            b"    alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [3]


def test_conditional_assignment_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"import os\n"
            b"from engine.v2.models import production_release_root\n"
            b"alias = production_release_root if os.name else None\n"
            b"alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [4]


def test_loop_binding_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root\n"
            b"for alias in (production_release_root,):\n"
            b"    alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [3]


def test_tuple_unpacking_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"import os\n"
            b"from engine.v2.models import production_release_root\n"
            b"first, second = production_release_root, os.name\n"
            b"first()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [4]


def test_with_binding_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root\n"
            b"with production_release_root as alias:\n"
            b"    alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [3]


def test_attribute_assignment_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import deployment\n"
            b"class Job:\n"
            b"    def run(self):\n"
            b"        self.resolver = deployment.production_release_root\n"
            b"        self.resolver()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [5]


def test_attribute_alias_resolves_across_methods():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import deployment\n"
            b"class Job:\n"
            b"    def __init__(self):\n"
            b"        self.resolver = deployment.production_release_root\n"
            b"    def run(self):\n"
            b"        self.resolver()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [6]


def test_getattr_assignment_alias_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import deployment\n"
            b"def run():\n"
            b"    resolver = getattr(deployment, 'production_release_root')\n"
            b"    resolver()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [4]


def test_getattr_call_is_enumerated():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import deployment\n"
            b"getattr(deployment, 'production_release_root')()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [2]


def test_experiment_star_import_exposing_live_ledger_is_refused():
    findings = isolation.check_files({
        "experiments/star.py": b"from engine.paths import *\np = LEDGER\n",
    })
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_experiment_qualified_live_release_attribute_is_refused():
    findings = isolation.check_files({
        "experiments/qualified.py": (
            b"import engine.v2.models.no_fit\n"
            b"engine.v2.models.current_release(None)\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_permitted_no_fit_submodule_import_remains_clean():
    assert isolation.check_files({
        "experiments/no_fit.py": (
            b"import engine.v2.models.no_fit\n"
            b"engine.v2.models.no_fit.prepare()\n"
        ),
    }) == []


@pytest.mark.parametrize(
    "body",
    [
        b"from engine.v2.models import no_fit\n",
        b"from engine.v2.models import fitting_forbidden\n",
    ],
)
def test_experiments_may_import_explicit_safe_models_symbols(body):
    assert isolation.check_files({"experiments/safe_symbol.py": body}) == []


def test_experiments_cannot_import_listed_models_symbol():
    findings = isolation.check_files({
        "experiments/unsafe_symbol.py":
            b"from engine.v2.models import deployment\n",
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [1]


@pytest.mark.parametrize(
    "body",
    [
        b"import engine.v2.models\n",
        b"from engine.v2 import models\n",
        b"from engine.v2.models import *\n",
    ],
)
def test_experiments_cannot_import_models_package_namespace(body):
    findings = isolation.check_files({"experiments/namespace.py": body})
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"import importlib\n"
        b"importlib.import_module('engine.v2.models.deployment')\n",
        b"import importlib as il\n"
        b"il.import_module('engine.v2.models.deployment')\n",
        b"from importlib import import_module as load\n"
        b"load('engine.v2.models.deployment')\n",
        b"__import__('engine.v2.models.deployment')\n",
    ],
)
def test_experiment_literal_dynamic_import_of_live_module_is_refused(body):
    findings = isolation.check_files({"experiments/dynamic.py": body})
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_dynamic_import_keyword_name_is_checked():
    findings = isolation.check_files({
        "experiments/dynamic_keyword.py": (
            b"import importlib\n"
            b"importlib.import_module(name='engine.v2.models.deployment')\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_dynamic_relative_import_with_literal_package_is_checked():
    findings = isolation.check_files({
        "experiments/dynamic_relative.py": (
            b"import importlib\n"
            b"importlib.import_module('.deployment', package='engine.v2.models')\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_dynamic_double_relative_import_resolves_by_level():
    findings = isolation.check_files({
        "experiments/dynamic_double_relative.py": (
            b"import importlib\n"
            b"importlib.import_module('..deployment', "
            b"package='engine.v2.models.releases')\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_dynamic_dunder_import_fromlist_member_is_checked():
    findings = isolation.check_files({
        "experiments/dynamic_fromlist.py":
            b"__import__('engine.v2', fromlist=['ledger'])\n",
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [1]


def test_dynamic_dunder_import_star_fromlist_exposing_descendants_is_refused():
    findings = isolation.check_files({
        "experiments/dynamic_star_fromlist.py":
            b"__import__('engine.v2', fromlist=['*'])\n",
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [1]


def test_unresolved_dynamic_import_target_is_refused():
    findings = isolation.check_files({
        "experiments/dynamic_unresolved.py": (
            b"import importlib\n"
            b"importlib.import_module(live_module)\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_relative_dynamic_import_without_literal_package_is_refused():
    findings = isolation.check_files({
        "experiments/dynamic_no_package.py": (
            b"import importlib\n"
            b"importlib.import_module('.deployment')\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_aliased_dynamic_import_is_checked():
    findings = isolation.check_files({
        "experiments/dynamic_alias.py": (
            b"from importlib import import_module as load\n"
            b"load(name='engine.v2.ledger.rows')\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


@pytest.mark.parametrize(
    "body",
    [
        b"import builtins as b\n"
        b"b.__import__('engine.v2.models.deployment')\n",
        b"from builtins import __import__ as load\n"
        b"load('engine.v2.models.deployment')\n",
    ],
)
def test_builtins_aliased_dynamic_import_is_checked(body):
    findings = isolation.check_files({"experiments/dynamic_builtins.py": body})
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_dynamic_models_namespace_attribute_is_refused():
    findings = isolation.check_files({
        "experiments/dynamic_models_attribute.py": (
            b"import importlib\n"
            b"importlib.import_module('engine.v2.models').current_release(None)\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_production_phase5_dynamic_holdout_import_is_refused():
    findings = isolation.check_files({
        "tools/phase5_runner.py": (
            b"import importlib\n"
            b"importlib.import_module('engine.v2.foundation.final_holdout')\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert [item.line for item in findings] == [2]


def test_unrelated_literal_dynamic_import_remains_clean():
    assert isolation.check_files({
        "tools/phase5_runner.py": (
            b"import importlib\n"
            b"importlib.import_module('json')\n"
        ),
    }) == []


def test_safe_literal_dynamic_no_fit_import_remains_clean():
    assert isolation.check_files({
        "experiments/dynamic_no_fit.py": (
            b"import importlib\n"
            b"importlib.import_module('engine.v2.models.no_fit')\n"
        ),
    }) == []


def test_conflicting_resolver_assignments_in_one_scope_do_not_oscillate():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root\n"
            b"from engine.v2.models import production_deployment_root\n"
            b"alias = production_release_root\n"
            b"alias = production_deployment_root\n"
            b"alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [5]


def test_conflicting_resolver_assignments_in_separate_scopes_do_not_oscillate():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root\n"
            b"from engine.v2.models import production_deployment_root\n"
            b"def first():\n"
            b"    alias = production_release_root\n"
            b"    alias()\n"
            b"def second():\n"
            b"    alias = production_deployment_root\n"
            b"    alias()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [5, 8]


def test_resolver_aliases_do_not_leak_between_functions():
    assert isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"def first():\n"
            b"    from engine.v2.models import production_release_root as alias\n"
            b"    alias = alias\n"
            b"def second():\n"
            b"    alias()\n"
        ),
    }) == []


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


@pytest.mark.parametrize(
    "body",
    [
        b"from engine.v2.scoring.release_bindings import "
        b"resolve_production_release_binding\n",
        b"from engine.v2.ops.pool_refresh import deployed_release_id\n",
        b"import engine.v2.models.serving_fold_preflight\n",
    ],
)
def test_experiment_facade_imports_are_refused(body):
    findings = isolation.check_files({"experiments/facade.py": body})
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_experiment_module_chain_facade_call_is_refused():
    findings = isolation.check_files({
        "experiments/facade_chain.py": (
            b"import engine.v2.scoring as scoring\n"
            b"scoring.release_bindings.resolve_production_release_binding()\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


@pytest.mark.parametrize(
    "body",
    [
        b"import importlib\n"
        b"load = importlib.import_module\n"
        b"load('engine.v2.models.deployment')\n",
        b"load = __import__\n"
        b"load('engine.v2.ledger')\n",
    ],
)
def test_importer_alias_dynamic_import_is_refused(body):
    findings = isolation.check_files({"experiments/importer_alias.py": body})
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_parameter_shadowing_importer_alias_remains_clean():
    assert isolation.check_files({
        "experiments/shadowed_importer.py": (
            b"import importlib\n"
            b"load = importlib.import_module\n"
            b"def run(load):\n"
            b"    load('engine.v2.models.deployment')\n"
        ),
    }) == []


def test_unlisted_root_calls_in_defaults_annotations_and_comprehension_are_enumerated():
    findings = isolation.check_files({
        "experiments/root_caller.py": (
            b"from engine.v2.models import production_release_root as root\n"
            b"def run(resolve=root): resolve()\n"
            b"callback = lambda resolve=root: resolve()\n"
            b"def annotated(value: root()): pass\n"
            b"def returned() -> root(): pass\n"
            b"[resolve() for resolve in (root,)]\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert len(findings) == 5


def test_tuple_alias_call_is_enumerated_but_attribute_and_lambda_are_clean():
    findings = isolation.check_files({
        "engine/v2/ops/new_job.py": (
            b"from engine.v2.models import production_release_root as resolve\n"
            b"service.resolve()\n"
            b"first, second = resolve, lambda: None\n"
            b"second()\n"
            b"first()\n"
        ),
    })
    assert _rules(findings) == {"unlisted-root-caller"}
    assert [item.line for item in findings] == [5]


def test_module_alias_chain_to_live_release_is_refused():
    findings = isolation.check_files({
        "experiments/module_chain.py": (
            b"import engine.v2.models.no_fit\n"
            b"m = engine.v2.models\n"
            b"m2 = m\n"
            b"m2.current_release(None)\n"
        ),
    })
    assert _rules(findings) == {"forbidden-import"}
    assert len(findings) == 1


def test_module_alias_no_fit_member_call_remains_clean():
    assert isolation.check_files({
        "experiments/module_alias_no_fit.py": (
            b"import engine.v2.models.no_fit\n"
            b"m = engine.v2.models\n"
            b"m.no_fit.prepare()\n"
        ),
    }) == []
