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


class _NoDuplicateKeysLoader(yaml.SafeLoader):
    """``yaml.safe_load`` silently keeps the LAST value of a duplicate
    mapping key, so a duplicate ``reviews.path_filters`` or
    ``reviews.path_instructions`` key could replace review policy while
    every test here still passes against the ``safe_load`` view. This
    loader raises instead."""


def _construct_mapping_no_duplicates(loader, node, deep=False):
    seen = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise ValueError(f"duplicate mapping key: {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_NoDuplicateKeysLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping_no_duplicates)


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
        return yaml.load(fh, Loader=_NoDuplicateKeysLoader)


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


def test_wildcard_path_instructions_state_strict_scope_rule():
    config = _load_config()
    path_instructions = config["reviews"]["path_instructions"]
    wildcard = next(e for e in path_instructions if e["path"] == "**/*")
    text = wildcard["instructions"]
    assert "Out of scope (suggest an issue)" in text
    assert "CHANGES_REQUESTED" in text


def test_wildcard_path_instructions_state_legacy_doc_exemption():
    # A legacy component (engine/** outside engine/v2/**) never gets a new
    # ARCHITECTURE.md; that must not be flaggable as a missing-doc defect.
    config = _load_config()
    path_instructions = config["reviews"]["path_instructions"]
    wildcard = next(e for e in path_instructions if e["path"] == "**/*")
    text = wildcard["instructions"]
    assert "engine/v2/**" in text
    assert "gets no new `ARCHITECTURE.md`" in text
    assert "PR body carries the design" in text


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
    # Use an untracked probe path, not the tracked template file: `git
    # check-ignore` reports a tracked path as not-ignored regardless of the
    # allow rule, so a tracked-file assertion here would pass vacuously.
    assert not _is_ignored("docs/coderabbit_allow_rule_probe.md")


def test_config_has_no_duplicate_mapping_keys():
    # A duplicate top-level or nested key would silently win under
    # yaml.safe_load; _load_config uses the strict loader above instead.
    _load_config()


def test_strict_loader_rejects_duplicate_keys():
    # Negative control: prove the strict loader actually catches a
    # duplicate key, so test_config_has_no_duplicate_mapping_keys above
    # is a real check and not a loader that quietly accepts everything.
    duplicated = "reviews:\n  profile: assertive\n  profile: chill\n"
    try:
        yaml.load(duplicated, Loader=_NoDuplicateKeysLoader)
    except ValueError:
        pass
    else:
        raise AssertionError("strict loader should have rejected a duplicate key")
