"""Structural checks that the root ARCHITECTURE.md 'Component docs' index
stays in sync with the real **/ARCHITECTURE.md tree, and that every sibling
README links its own ARCHITECTURE.md (spec_architecture_docs).

Tier 0: walks the checked-out tree and parses markdown link syntax as text
only. No network, no data, no fitting.
"""
from __future__ import annotations
# land: always-run

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_ARCHITECTURE = ROOT / "ARCHITECTURE.md"

# Matches a markdown link's target: [text](target)
_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")

_EXCLUDED_PARTS = {".git", "node_modules"}


def _excluded(path: Path) -> bool:
    return any(part in _EXCLUDED_PARTS for part in path.parts)


def _extract_links(text: str) -> list[str]:
    return _LINK_RE.findall(text)


def _non_root_architecture_docs(root: Path) -> list[Path]:
    root_doc = root / "ARCHITECTURE.md"
    docs = []
    for path in root.glob("**/ARCHITECTURE.md"):
        if _excluded(path) or path.resolve() == root_doc.resolve():
            continue
        docs.append(path)
    return docs


def _root_index_linked_targets(root: Path) -> set[Path]:
    root_doc = root / "ARCHITECTURE.md"
    text = root_doc.read_text(encoding="utf-8")
    targets = set()
    for link in _extract_links(text):
        if link.startswith(("http://", "https://")):
            continue
        targets.add((root / link.split("#")[0]).resolve())
    return targets


def _unlinked_non_root_docs(root: Path) -> list[str]:
    linked = _root_index_linked_targets(root)
    missing = []
    for doc in _non_root_architecture_docs(root):
        if doc.resolve() not in linked:
            missing.append(str(doc.relative_to(root)))
    return sorted(missing)


def _broken_root_index_links(root: Path) -> list[str]:
    root_doc = root / "ARCHITECTURE.md"
    text = root_doc.read_text(encoding="utf-8")
    broken = []
    for link in _extract_links(text):
        if link.startswith(("http://", "https://", "#")):
            continue
        target = (root / link.split("#")[0]).resolve()
        if not target.is_file():
            broken.append(link)
    return sorted(broken)


def _readmes_missing_architecture_link(root: Path) -> list[str]:
    missing = []
    for arch_doc in root.glob("**/ARCHITECTURE.md"):
        if _excluded(arch_doc):
            continue
        readme = arch_doc.parent / "README.md"
        if not readme.is_file():
            continue
        text = readme.read_text(encoding="utf-8")
        resolved_targets = set()
        for link in _extract_links(text):
            if link.startswith(("http://", "https://")):
                continue
            resolved_targets.add((readme.parent / link.split("#")[0]).resolve())
        if arch_doc.resolve() not in resolved_targets:
            missing.append(str(readme.relative_to(root)))
    return sorted(missing)


def test_every_non_root_architecture_doc_is_linked_from_root_index():
    missing = _unlinked_non_root_docs(ROOT)
    assert not missing, (
        "these ARCHITECTURE.md files are not linked from the root "
        f"ARCHITECTURE.md 'Component docs' index: {missing}"
    )


def test_every_root_index_link_resolves():
    broken = _broken_root_index_links(ROOT)
    assert not broken, (
        f"these links in the root ARCHITECTURE.md do not resolve to a real file: {broken}"
    )


def test_every_sibling_readme_links_its_architecture_doc():
    missing = _readmes_missing_architecture_link(ROOT)
    assert not missing, (
        f"these README.md files do not link their sibling ARCHITECTURE.md: {missing}"
    )


def test_check_catches_an_unlinked_architecture_doc(tmp_path):
    # Negative control: prove _unlinked_non_root_docs actually flags a real
    # gap, not just passes on the real tree by construction.
    (tmp_path / "ARCHITECTURE.md").write_text("# Architecture\n\nNo links here.\n",
                                               encoding="utf-8")
    component = tmp_path / "componentX"
    component.mkdir()
    (component / "ARCHITECTURE.md").write_text("# X\n", encoding="utf-8")
    assert _unlinked_non_root_docs(tmp_path) == ["componentX/ARCHITECTURE.md"]


def test_check_catches_a_broken_root_index_link(tmp_path):
    # Negative control: prove _broken_root_index_links actually flags a
    # dangling link and does not flag a real one, using fully synthetic
    # content isolated from the real repo's own ARCHITECTURE.md.
    component = tmp_path / "componentZ"
    component.mkdir()
    (component / "ARCHITECTURE.md").write_text("# Z\n", encoding="utf-8")
    text = ("# Architecture\n\n"
            "[good](componentZ/ARCHITECTURE.md)\n"
            "[bad](does/not/exist/ARCHITECTURE.md)\n")
    (tmp_path / "ARCHITECTURE.md").write_text(text, encoding="utf-8")
    assert _broken_root_index_links(tmp_path) == ["does/not/exist/ARCHITECTURE.md"]


def test_check_catches_a_readme_missing_its_architecture_link(tmp_path):
    # Negative control: prove _readmes_missing_architecture_link actually
    # flags a README with no link to its sibling doc.
    component = tmp_path / "componentY"
    component.mkdir()
    (component / "ARCHITECTURE.md").write_text("# Y\n", encoding="utf-8")
    (component / "README.md").write_text("# componentY\n\nNo link here.\n", encoding="utf-8")
    assert _readmes_missing_architecture_link(tmp_path) == ["componentY/README.md"]
