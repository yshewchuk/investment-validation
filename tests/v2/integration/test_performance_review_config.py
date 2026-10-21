"""Shared review guidance reaches the native workload review entrypoints.

# land: always-run
# packages: engine.v2.ops, engine.v2.research, engine.v2.scoring
"""
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
POLICY_LIMIT = 256 * 1024


def read_policy(path):
    """Bound these small test policy inputs, not production source readers."""
    with path.open("rb") as stream:
        payload = stream.read(POLICY_LIMIT + 1)
    assert len(payload) <= POLICY_LIMIT, f"policy test input exceeds {POLICY_LIMIT} bytes: {path}"
    return payload.decode("utf-8")


def test_performance_guide_is_loaded_and_referenced():
    config = yaml.safe_load(read_policy(ROOT / ".coderabbit.yaml"))
    guidelines = config["knowledge_base"]["code_guidelines"]
    assert guidelines["enabled"] is True
    assert "docs/performance_review.md" in guidelines["filePatterns"]
    wildcard = next(entry for entry in config["reviews"]["path_instructions"]
                    if entry["path"] == "**/*")
    assert "/docs/performance_review.md" in wildcard["instructions"]
    assert "(docs/performance_review.md)" in read_policy(ROOT / "AGENTS.md")
    guide = " ".join(read_policy(ROOT / "docs/performance_review.md").split())
    for criterion in (
        "its caller/loop and plausible input size or call count before raising a finding",
        "repeated full-inventory scans or identical pinned reads",
        "holiday calendars; repeated sorting, JSON normalization or array conversion",
        "not permission to skip fresh integrity checks on later replay or recovery calls",
        "verify actual immutable identity, not just a caller-declared hash",
        "separate snapshots, sessions and requests as required",
        "Check entry/byte bounds, retained-object memory, mutation safety, eviction and concurrency",
        "cold-start validation/allocation costs and realistic reuse frequency",
        "not extrapolate a helper speedup into a whole-nightly or whole-experiment claim",
        "Performance is not a blanket reason to block a PR",
        "deterministic read/call-count tests",
    ):
        assert criterion in guide


def test_policy_reader_accepts_limit_and_refuses_larger_input(tmp_path):
    policy = tmp_path / "policy.md"
    payload = ("aéZ\n".encode("utf-8") * (POLICY_LIMIT // 5)
               + b"!" * (POLICY_LIMIT % 5))
    policy.write_bytes(payload)
    assert read_policy(policy) == payload.decode("utf-8")
    with policy.open("ab") as stream:
        stream.write(b"b")
    with pytest.raises(AssertionError, match="policy test input exceeds 262144 bytes"):
        read_policy(policy)
