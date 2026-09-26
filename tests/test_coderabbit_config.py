"""Structural checks for .coderabbit.yaml and the ARCHITECTURE.md docs
(spec_coderabbit_config).

Tier 0: parses the checked-in config, the repo tree and .gitignore only. No
network, no data, no fitting.
"""
from __future__ import annotations
# land: always-run

import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".coderabbit.yaml"


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path],
        cwd=ROOT, check=False,
    )
    return result.returncode == 0


def test_config_parses_as_a_mapping():
    config = _load_config()
    assert isinstance(config, dict)


def test_code_guidelines_load_architecture_docs():
    config = _load_config()
    patterns = config["knowledge_base"]["code_guidelines"]["filePatterns"]
    assert "**/ARCHITECTURE.md" in patterns


def test_path_instructions_globs_match_existing_files():
    config = _load_config()
    path_instructions = config["reviews"]["path_instructions"]
    assert path_instructions, "expected at least one path_instructions entry"
    for entry in path_instructions:
        pattern = entry["path"]
        matches = [p for p in ROOT.glob(pattern) if p.is_file()]
        assert matches, f"path_instructions pattern {pattern!r} matches no existing file"


def test_gitignore_admits_architecture_md_anywhere():
    # Root and nested component docs must be trackable...
    for rel in ("ARCHITECTURE.md", "engine/v2/ops/ARCHITECTURE.md",
               "engine/v2/dashboard/ARCHITECTURE.md",
               "engine/dashboard/ARCHITECTURE.md",
               "brand_new_component/ARCHITECTURE.md"):
        assert not _is_ignored(rel), f"{rel} should be admitted by .gitignore"
    # ...but the same rule must not open the door to an unrelated file with
    # a different name in a directory that has no allowlist of its own.
    for rel in ("brand_new_component/NOTES.md", "brand_new_component/README.md"):
        assert _is_ignored(rel), f"{rel} should still be blocked by the default-deny"


def test_docs_dir_still_admits_plain_markdown():
    assert not _is_ignored("docs/COMPONENT_ARCHITECTURE_TEMPLATE.md")
