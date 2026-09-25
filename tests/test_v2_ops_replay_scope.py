"""The tier-1 receipt's code hash covers what a replay executes, and only that.

Rearchitecture phase 1 decision D6 narrowed ``checks.replay_identity.code_hash``
from all of ``engine/**`` to legacy ``engine/`` plus the v2 packages the replay
harness actually imports. A narrowing is only safe while it stays true, so the
closure is re-derived here from the import graph rather than trusted.
"""
from __future__ import annotations
# land: always-run

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from checks import replay_identity as identity  # noqa: E402
from checks.import_layers import imports_in, module_name  # noqa: E402
from checks.layer_map import package_of  # noqa: E402

HARNESS = (*identity.CODE_FILES,)


def v2_closure(files: dict[str, bytes], seeds: tuple[str, ...]) -> set[str]:
    """v2 package paths reachable by import from ``seeds`` through ``files``."""
    by_package: dict[str, list[str]] = {}
    for rel in files:
        pkg = package_of(module_name(rel) or "")
        if pkg is not None:
            by_package.setdefault(pkg.path, []).append(rel)
    reached: set[str] = set()
    frontier = list(seeds)
    while frontier:
        rel = frontier.pop()
        importer = module_name(rel) or rel
        for edge in imports_in(files[rel], rel, importer):
            pkg = package_of(edge.imported)
            if pkg is not None and pkg.path not in reached:
                reached.add(pkg.path)
                frontier.extend(by_package.get(pkg.path, []))
    return reached


def _tree_files() -> dict[str, bytes]:
    files = {p.relative_to(ROOT).as_posix(): p.read_bytes()
             for p in (ROOT / "engine" / "v2").rglob("*.py") if "__pycache__" not in p.parts}
    files.update({rel: (ROOT / rel).read_bytes() for rel in HARNESS})
    return files


def test_declared_scope_covers_every_v2_package_the_harness_reaches():
    reached = v2_closure(_tree_files(), HARNESS)
    assert reached, "the harness imports diagnosis; an empty closure means the walk is broken"
    assert reached <= set(identity.REPLAY_V2_PACKAGES), sorted(reached)


def test_a_harness_importing_ops_would_be_caught():
    """Negative control: plant the import the narrowing depends on never existing."""
    files = _tree_files()
    files["checks/tier0_corpus.py"] += b"\nfrom engine.v2.ops import catalog\n"
    files["engine/v2/ops/catalog.py"] = b""
    reached = v2_closure(files, HARNESS)
    assert "engine/v2/ops" in reached
    assert not reached <= set(identity.REPLAY_V2_PACKAGES)


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_ops_edit_does_not_move_the_hash_but_legacy_and_diagnosis_edits_do(tmp_path):
    for rel in ("engine/score.py", "engine/v2/__init__.py", "engine/v2/ops/catalog.py",
                "engine/v2/diagnosis/receipt.py", "engine/v2/foundation/canonical.py"):
        _write(tmp_path, rel, "x = 1\n")
    base = identity.code_hash(tmp_path)

    _write(tmp_path, "engine/v2/ops/catalog.py", "x = 2\n")
    assert identity.code_hash(tmp_path) == base

    for rel in ("engine/score.py", "engine/v2/diagnosis/receipt.py",
                "engine/v2/foundation/canonical.py"):
        _write(tmp_path, rel, "x = 3\n")
        moved = identity.code_hash(tmp_path)
        assert moved != base, rel
        base = moved
