"""Negative controls for the §4.3 budgets and the §4.5 README check.

Both checks are proved by planting what they exist to catch. Asserting only
that the real tree is green would pass identically if either check did nothing.
"""
from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import code_budgets as cb  # noqa: E402
from checks import install_hooks, package_readmes as pr  # noqa: E402
from checks.layer_map import PACKAGES, README_SECTIONS  # noqa: E402


def metrics(report) -> set[str]:
    return {v.metric for v in report.violations}


def blocking(report) -> set[str]:
    return {v.metric for v in report.blocking}


# --------------------------------------------------------------------------
# scope
# --------------------------------------------------------------------------


def test_the_real_v2_tree_is_within_budget():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "code_budgets.py"), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def test_legacy_is_exempt_wholesale():
    """§4.6: legacy is frozen and deleted at phase 8; taxing it buys nothing."""
    assert not cb.in_scope("engine/score.py")
    assert not cb.in_scope("checks/repo_hygiene.py")
    assert cb.in_scope("engine/v2/scoring/kernel.py")
    monster = b"def f():\n" + b"    if x:\n        pass\n" * 60
    assert cb.check_files({"engine/score.py": monster}).violations == []
    assert cb.check_files({"engine/v2/scoring/k.py": monster}).violations != []


# --------------------------------------------------------------------------
# the four budgets
# --------------------------------------------------------------------------


def test_a_function_over_complexity_fails():
    body = "\n".join(f"    if x == {i}:\n        return {i}" for i in range(20))
    src = f"def f(x):\n{body}\n    return None\n".encode()
    assert "complexity" in blocking(cb.check_files({"engine/v2/ops/j.py": src}))


def test_a_function_over_eighty_lines_fails():
    src = ("def f():\n" + "    a = 1\n" * 90).encode()
    report = cb.check_files({"engine/v2/ops/j.py": src})
    assert "function_lines" in blocking(report)


def test_a_module_over_six_hundred_lines_warns_but_does_not_block():
    """§4.3 states module length as a soft cap enforced as a warning."""
    src = ("x = 1\n" * 700).encode()
    report = cb.check_files({"engine/v2/ops/j.py": src})
    assert "module_lines" in metrics(report)
    assert "module_lines" not in blocking(report)
    assert report.ok


def test_fan_out_above_eight_fails_for_a_non_orchestrator():
    imports = "\n".join(f"import mod{i}" for i in range(12))
    src = imports.encode()
    assert "fan_out" in blocking(cb.check_files({"engine/v2/features/f.py": src}))


def test_an_orchestrator_is_exempt_from_fan_out_and_not_from_length():
    """§4.1/§4.3: an orchestrator is exactly where a failure must localize."""
    imports = "\n".join(f"import mod{i}" for i in range(12)).encode()
    assert cb.check_files({"engine/v2/scoring/kernel.py": imports}).violations == []
    long_fn = ("def run():\n" + "    a = 1\n" * 90).encode()
    assert "function_lines" in blocking(
        cb.check_files({"engine/v2/scoring/kernel.py": long_fn})
    )


def test_future_import_is_not_counted_as_a_dependency():
    src = "from __future__ import annotations\nimport json\n"
    assert cb.fan_out(ast.parse(src)) == {"json"}


def test_complexity_counts_each_boolean_operand():
    one = cb.complexity(ast.parse("def f(a,b,c):\n    return a and b and c").body[0])
    plain = cb.complexity(ast.parse("def f(a):\n    return a").body[0])
    assert one - plain == 2


def test_a_nested_function_is_not_counted_twice():
    src = textwrap.dedent("""
        def outer(x):
            def inner(y):
                if y:
                    return 1
                return 2
            return inner(x)
    """)
    outer = ast.parse(src).body[0]
    assert cb.complexity(outer) == 1


def test_there_is_no_exemption_mechanism():
    """§4.6: zero tolerance, no exemption file, no grandfathering.

    Proved by there being nothing to prove it against: the check takes files
    and nothing else, its CLI offers no way to skip one, and no exemption file
    exists for it to read. An exemption list is a one-line edit away in any
    design that has a slot for it, which is why this one has none.
    """
    import inspect

    assert set(inspect.signature(cb.check_files).parameters) == {"files"}
    cli = (ROOT / "checks" / "code_budgets.py").read_text()
    for flag in ("--exempt", "--ignore", "--skip", "--allow"):
        assert flag not in cli
    assert not list(ROOT.glob("checks/*exempt*"))


# --------------------------------------------------------------------------
# READMEs
# --------------------------------------------------------------------------


def readme(**overrides) -> str:
    sections = "\n\n".join(f"## {name}\n\nbody" for name in README_SECTIONS)
    consumers = overrides.get("consumers", "none")
    interface = overrides.get("interface", "none")
    return (f"# pkg\n\n{sections}\n\n"
            f"<!-- consumers: {consumers} -->\n"
            f"<!-- public-interface: {interface} -->\n")


def test_the_real_tree_passes_the_readme_check():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "checks" / "package_readmes.py"), "--all"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr


def test_every_package_has_a_readme_with_all_seven_sections():
    for pkg in PACKAGES:
        text = (ROOT / pkg.path / "README.md").read_text()
        assert pr.missing_sections(text) == [], pkg.path


def test_a_missing_readme_fails():
    readmes = {p.dotted: readme() for p in PACKAGES}
    readmes["engine.v2.ledger"] = None
    report = pr.check({}, readmes)
    assert any(v.rule == "missing-readme" for v in report.violations)


def test_a_missing_section_fails():
    readmes = {p.dotted: readme() for p in PACKAGES}
    readmes["engine.v2.ledger"] = readme().replace("## Consumers", "## Users")
    report = pr.check({}, readmes)
    assert any(v.rule == "missing-section" for v in report.violations)


def test_a_claimed_consumer_that_does_not_import_fails():
    readmes = {p.dotted: readme() for p in PACKAGES}
    readmes["engine.v2.features"] = readme(consumers="engine.v2.scoring")
    report = pr.check({}, readmes)
    assert any(v.rule == "consumer-not-observed" for v in report.violations)


def test_an_omitted_consumer_that_does_import_fails():
    readmes = {p.dotted: readme() for p in PACKAGES}
    files = {"engine/v2/scoring/kernel.py": b"from engine.v2.features import f\n"}
    report = pr.check(files, readmes)
    assert any(v.rule == "consumer-not-declared" for v in report.violations)


def test_a_declared_consumer_that_does_import_passes():
    readmes = {p.dotted: readme() for p in PACKAGES}
    readmes["engine.v2.features"] = readme(consumers="engine.v2.scoring",
                                           interface="f")
    files = {"engine/v2/scoring/kernel.py": b"from engine.v2.features import f\n"}
    report = pr.check(files, readmes)
    assert report.violations == [], [str(v) for v in report.violations]


def test_importing_a_name_outside_the_public_interface_fails():
    readmes = {p.dotted: readme() for p in PACKAGES}
    readmes["engine.v2.features"] = readme(consumers="engine.v2.scoring")
    files = {"engine/v2/scoring/kernel.py": b"from engine.v2.features import f\n"}
    report = pr.check(files, readmes)
    assert any(v.rule == "private-name-imported" for v in report.violations)


def test_an_absent_directive_is_not_the_same_as_an_empty_one():
    readmes = {p.dotted: readme() for p in PACKAGES}
    sections = "\n\n".join(f"## {n}\n\nbody" for n in README_SECTIONS)
    readmes["engine.v2.ledger"] = f"# pkg\n\n{sections}\n"
    report = pr.check({}, readmes)
    rules = {v.rule for v in report.violations}
    assert {"undeclared-consumers", "undeclared-interface"} <= rules


@pytest.mark.parametrize("raw,expected", [
    ("<!-- consumers: none -->", set()),
    ("<!-- consumers: a, b -->", {"a", "b"}),
    ("<!-- consumers:  -->", set()),
    ("nothing here", None),
])
def test_directive_parsing(raw, expected):
    assert pr.directive(raw, "consumers") == expected


# --------------------------------------------------------------------------
# the hook
# --------------------------------------------------------------------------


def test_the_versioned_hook_exists_and_runs_every_check():
    source = (ROOT / "checks" / "hooks" / "pre-commit").read_text()
    for check in ("repo_hygiene.py", "import_layers.py", "code_budgets.py",
                  "package_readmes.py"):
        assert check in source


def test_the_hook_is_installed_and_current():
    """§4.6: a missing hook is a failure on the same footing as a budget one."""
    state = install_hooks.state(ROOT)
    assert state["installed"], state
    assert state["current"], state
    assert state["executable"], state


def test_install_refuses_to_clobber_an_unrecognized_hook(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "checks" / "hooks").mkdir(parents=True)
    (tmp_path / "checks" / "hooks" / "pre-commit").write_text("#!/bin/sh\ntrue\n")
    hooks = install_hooks.hooks_dir(tmp_path)
    hooks.mkdir(parents=True, exist_ok=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\n# somebody else's hook\n")
    result = install_hooks.install(tmp_path)
    assert result["written"] is False
    assert "force" in result["reason"]
    assert install_hooks.install(tmp_path, force=True)["written"] is True
