"""Shared review guidance reaches the native workload review entrypoints.

# land: always-run
# packages: engine.v2.ops, engine.v2.research, engine.v2.scoring
"""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def test_performance_guide_is_loaded_and_referenced():
    config = yaml.safe_load((ROOT / ".coderabbit.yaml").read_text())
    guidelines = config["knowledge_base"]["code_guidelines"]
    assert guidelines["enabled"] is True
    assert "docs/performance_review.md" in guidelines["filePatterns"]
    wildcard = next(entry for entry in config["reviews"]["path_instructions"]
                    if entry["path"] == "**/*")
    assert "/docs/performance_review.md" in wildcard["instructions"]
    assert "(docs/performance_review.md)" in (ROOT / "AGENTS.md").read_text()
    assert (ROOT / "docs/performance_review.md").is_file()
