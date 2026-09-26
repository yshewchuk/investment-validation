"""Structural checks for .coderabbit.yaml (spec_coderabbit_config).

Tier 0: parses the checked-in config and the repo tree only. No network, no
data, no fitting.
"""
from __future__ import annotations
# land: always-run

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / ".coderabbit.yaml"


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_config_parses_as_a_mapping():
    config = _load_config()
    assert isinstance(config, dict)


def test_code_guidelines_load_architecture_doc():
    config = _load_config()
    patterns = config["knowledge_base"]["code_guidelines"]["filePatterns"]
    assert "docs/ARCHITECTURE.md" in patterns


def test_path_instructions_globs_match_existing_files():
    config = _load_config()
    path_instructions = config["reviews"]["path_instructions"]
    assert path_instructions, "expected at least one path_instructions entry"
    for entry in path_instructions:
        pattern = entry["path"]
        matches = [p for p in ROOT.glob(pattern) if p.is_file()]
        assert matches, f"path_instructions pattern {pattern!r} matches no existing file"
