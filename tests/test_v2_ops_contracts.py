"""The phase-1 operations contracts are schemas, and nothing else (slice P1-1).

``engine/v2/contracts`` may hold no logic, no I/O and no hashing computed during
construction (§4). Those rules are structural claims about source, so they are
checked against the source rather than trusted.
"""
from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from typing import get_args

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.v2 import contracts  # noqa: E402
from engine.v2.contracts import (  # noqa: E402
    FAILURE_CODES,
    ArtifactRef,
    CheckpointCandidate,
    CheckpointReceipt,
    JobSpec,
    LegacyFileRef,
    LegacyInputManifest,
    OutputCandidate,
    Problem,
    ProblemCategory,
    ResourcePolicy,
    ResourceProfile,
    StageResult,
    StageSpec,
    SubmitRequest,
)
from engine.v2.foundation import (  # noqa: E402
    DocumentError,
    from_document,
    parse_schema_version,
    to_document,
)

CONTRACT_FILES = sorted((ROOT / "engine" / "v2" / "contracts").glob("*.py"))
_ALLOWED_IMPORTS = {"__future__", "dataclasses", "typing"}


@pytest.mark.parametrize("path", CONTRACT_FILES, ids=lambda p: p.name)
def test_contracts_import_nothing_that_could_do_io(path):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert name in _ALLOWED_IMPORTS or name.startswith("engine.v2.contracts"), name


@pytest.mark.parametrize("path", CONTRACT_FILES, ids=lambda p: p.name)
def test_contracts_define_no_functions(path):
    """Schemas only: a method is where a hash-on-construction would hide."""
    tree = ast.parse(path.read_text())
    assert not [n.name for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))]


def _contract_types():
    return [obj for obj in vars(contracts).values()
            if isinstance(obj, type) and dataclasses.is_dataclass(obj)]


def test_every_versioned_contract_has_a_unique_well_formed_family():
    families = {}
    for cls in _contract_types():
        fields = {f.name: f for f in dataclasses.fields(cls)}
        if "schema_version" not in fields:
            continue
        family, major, _ = parse_schema_version(fields["schema_version"].default)
        assert major == 1
        assert family not in families, (cls, families.get(family))
        families[family] = cls
    assert {"job_spec", "attempt_receipt", "checkpoint_receipt", "stage_result",
            "problem", "progress_event", "resource_policy"} <= set(families)


def test_every_contract_is_frozen_and_keyword_only():
    for cls in _contract_types():
        params = cls.__dataclass_params__
        assert params.frozen, cls
        assert all(f.kw_only for f in dataclasses.fields(cls)), cls


def test_failure_codes_cover_section_5_3_with_valid_categories():
    required = {
        "RESOURCE_UNAVAILABLE", "RESOURCE_LIMIT_EXCEEDED", "TRANSIENT_SOURCE",
        "RATE_LIMITED", "CREDENTIAL_INVALID", "SOURCE_NOT_FOUND", "SOURCE_EMPTY",
        "SOURCE_NOT_FINAL", "INPUT_CHANGED", "CHECKPOINT_INCOMPATIBLE",
        "VALIDATION_FAILED", "INTEGRITY_FAILED", "LEASE_LOST", "CANCELLED",
        "PUBLICATION_REFUSED", "DELIVERY_FAILED", "BACKUP_FAILED",
        "IDEMPOTENCY_CONFLICT",
    }
    assert required <= set(FAILURE_CODES)
    categories = set(get_args(ProblemCategory))
    assert {cat for cat, _ in FAILURE_CODES.values()} <= categories
    # Credential and integrity failures must never default to automatic retry.
    assert FAILURE_CODES["CREDENTIAL_INVALID"][1] is False
    assert FAILURE_CODES["INTEGRITY_FAILED"][1] is False


def test_construction_computes_no_identity():
    spec = JobSpec(kind="k", implementation_ref="i", spec_hash=None, environment_ref="e",
                   output_namespace="ns", resource_class="io_fetch",
                   retry_policy_ref="r", checkpoint_contract_ref="c")
    assert spec.spec_hash is None


def _samples():
    ref = ArtifactRef(artifact_id="art_1", content_hash="sha256:" + "0" * 64,
                      schema_ref="s.v1.0", byte_size=1, storage_key="objects/00/" + "0" * 64)
    spec = JobSpec(kind="k", implementation_ref="i", spec_hash="sha256:" + "1" * 64,
                   environment_ref="e", output_namespace="ns", resource_class="legacy_score",
                   retry_policy_ref="r", checkpoint_contract_ref="c", priority=5,
                   input_refs=("in_b", "in_a"), parameters={"as_of": "2026-09-11"})
    output = OutputCandidate(name="scores", staged_path="batch_0/scores.json",
                             schema_ref="s.v1.0")
    return [
        ref,
        SubmitRequest(namespace="nightly", idempotency_key="n-2026-09-11",
                      principal="timer", job=spec),
        StageSpec(stage_id="score", job_kind="legacy.score_batch", implementation_ref="i",
                  parameter_ref="p", input_contract_ref="in", output_contract_ref="out",
                  dependency_stage_ids=("features",), resource_class="legacy_score",
                  retry_policy_ref="r", checkpoint_contract_ref="c", effect_class="staged",
                  required_validation_kinds=("coverage",), determinism_policy_ref="d"),
        LegacyInputManifest(manifest_id="m", file_refs=(LegacyFileRef(
            path="data/curated/x.parquet", content_hash=ref.content_hash, byte_size=9),),
            table_contract_refs=(), registry_and_model_refs=("registry",), calendar_ref=None,
            selected_session="2026-09-11", finality_receipt_refs=(),
            knowledge_mode_by_table={"x": "reconstructed"}, availability_evidence_refs=(),
            read_set_complete=False, capture_implementation_ref="cap"),
        CheckpointReceipt(stage_id="score", shard_key="batch_0", cache_key="ck",
                          input_hash="ih", implementation_hash="impl", parameter_hash="ph",
                          environment_hash="eh", output_schema_ref="s.v1.0",
                          artifact_refs=(ref,), validation_refs=(), producer_attempt_id="att",
                          producer_fence=3, committed_at="2026-09-12T00:00:00.000000Z"),
        StageResult(job_id="job", attempt_id="att", fence=3, stage_id="score",
                    input_manifest_ref="m", output_candidates=(output,),
                    checkpoint_candidates=(CheckpointCandidate(
                        shard_key="batch_0", cache_key="ck", input_hash="ih",
                        implementation_hash="impl", parameter_hash="ph",
                        environment_hash="eh", output_schema_ref="s.v1.0",
                        outputs=(output,), coverage={"expected": 3}),),
                    completion_counts={"accepted": 2, "refused": 1},
                    failure=Problem(code="VALIDATION_FAILED", category="validation",
                                    retryable=False, message="coverage short")),
        ResourcePolicy(version="policy.v1", base_reserve_bytes=1, free_margin_bytes=1,
                       reserved_cpu_count=1, min_free_disk_bytes=1, max_heavy_concurrency=1,
                       max_disk_heavy_concurrency=1, profiles=(ResourceProfile(
                           name="legacy_score", memory_bytes=3 << 30, cpu_count=4,
                           scratch_bytes=1 << 30, heavy=True),)),
    ]


@pytest.mark.parametrize("sample", _samples(), ids=lambda s: type(s).__name__)
def test_contracts_round_trip_exactly_and_preserve_array_order(sample):
    doc = to_document(sample)
    assert from_document(type(sample), doc) == sample


def test_effect_class_outside_the_vocabulary_is_refused():
    doc = to_document(_samples()[2])
    doc["effect_class"] = "side_effect_whatever"
    with pytest.raises(DocumentError) as err:
        from_document(StageSpec, doc)
    assert err.value.code == "BAD_ENUM"
