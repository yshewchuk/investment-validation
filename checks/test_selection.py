#!/usr/bin/env python3
"""Select v2 tests for changed paths: package ownership plus reverse-import closure."""
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from checks.layer_map import PACKAGES, package_of  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
V2, TESTS = "engine/v2/", "tests/v2/"
INTEGRATION = TESTS + "integration/"
META_TESTS = (
    "tests/test_import_layers.py", "tests/test_layer_map.py", "tests/test_repo_hygiene.py",
    "tests/test_architecture_docs.py", "tests/test_architecture_doc_budgets.py",
    "tests/test_code_budgets.py", INTEGRATION + "test_test_selection.py",
)
_FULL = ("checks/", "tools/", "ui/", ".github/")
_DECL = re.compile(r"^#\s*packages\s*:\s*(.*?)\s*$", re.MULTILINE)
NAMES = {name: pkg for pkg in PACKAGES
         for name in (pkg.dotted, pkg.path[len(V2):], pkg.path[len(V2):].replace("/", "."))}


@dataclass(frozen=True)
class Selection:
    targets: tuple[str, ...]
    full_suite: bool = False
    reason: str | None = None
    errors: tuple[str, ...] = ()


def full(reason, *errors):
    return Selection(("tests",), True, reason, tuple(errors))


def package_for(path, prefix):
    rel = path[len(prefix):] if path.startswith(prefix) else ""
    rel = rel[:-3] if rel.endswith(".py") else rel
    return package_of("engine.v2." + rel.replace("/", ".")) if rel else None


def read(root, path):
    try:
        return (root / path).read_text(errors="replace")
    except OSError:
        return None


def declaration(root, path, known):
    text = read(root, path)
    if text is None:
        return None, f"cannot read {path}"
    found = _DECL.findall(text)
    if len(found) != 1:
        return None, ("missing" if not found else "more than one") + " '# packages:' declaration"
    names = tuple(n.strip() for n in found[0].split(","))
    if not found[0].strip() or not all(names):
        return None, "empty '# packages:' declaration"
    if len(set(names)) != len(names):
        return None, "duplicate package in '# packages:' declaration"
    if any(n not in known for n in names):
        return None, f"unknown package {next(n for n in names if n not in known)!r}"
    return names, None


def _eligible(importer, dependency):
    if importer.sink:
        return True
    if importer.only_imports is None:
        return dependency.layer < importer.layer
    return dependency.layer in importer.only_imports


def closure(changed):
    selected = set(changed)
    while True:
        added = {p for p in PACKAGES if p not in selected
                 and any(_eligible(p, d) for d in selected)}
        if not added:
            return selected
        selected |= added


def select(changed, root=ROOT):
    root = Path(root)
    paths = sorted({re.sub(r"^(?:\./)+", "", str(p).replace("\\", "/")) for p in changed if p})
    if not paths:
        return full("no changed paths")
    changed, selected = set(paths), set()
    for path in paths:
        if path.startswith(_FULL):
            return full(f"{path} is outside the layer map")
        if (path == "ARCHITECTURE.md" or path.endswith("/ARCHITECTURE.md")
                or (path.startswith("guides/") and path.endswith(".md"))
                or path.startswith(INTEGRATION)):
            continue
        if path.startswith(TESTS) and _DECL.search(read(root, path) or ""):
            return full(f"misplaced '# packages:' declaration in {path}",
                        "declarations are only valid under " + INTEGRATION)
        pkg = package_for(path, V2) or package_for(path, TESTS)
        if pkg is None:
            return full(f"{path} does not resolve to a package")
        selected.add(pkg)
    directory, integration = root / INTEGRATION, {p for p in paths if p.startswith(INTEGRATION)}
    if directory.is_dir():
        integration |= {INTEGRATION + p.relative_to(directory).as_posix()
                        for p in directory.rglob("*.py") if p.name != "__init__.py"}
    declared = {}
    for path in sorted(integration):
        names, error = declaration(root, path, NAMES)
        if error is not None:
            return full(f"invalid integration declaration in {path}", error)
        declared[path] = [NAMES[n] for n in names]
        if path in changed:
            selected.update(declared[path])
    keep = closure(selected)
    targets = {TESTS + pkg.path[len(V2):] for pkg in keep} | set(META_TESTS)
    targets |= {p for p, pkgs in declared.items() if any(k in keep for k in pkgs)}
    return Selection(tuple(sorted(targets)))
