"""Immutable model artifact inventory and release completeness contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ARTIFACT_INVENTORY_MEMBER_V1 = "artifact_inventory_member.v1.0"
MODEL_ARTIFACT_INVENTORY_V1 = "model_artifact_inventory.v1.0"
RELEASE_BINDING_V1 = "model_release_binding.v1.0"
RELEASE_REQUIREMENT_V1 = "model_release_requirement.v1.0"
MODEL_RELEASE_V1 = "model_release.v1.0"
MODEL_RELEASE_REFUSAL = "MODEL_RELEASE_INCOMPLETE"

ModelRole = Literal["size", "implied_t1", "runup_move", "iv_crush", "gate", "chooser"]
MemberKind = Literal[
    "estimator", "transform", "residual", "residual_bucket",
    "calibration", "threshold", "paired_simulation",
]


@dataclass(frozen=True, kw_only=True)
class ArtifactInventoryMember:
    member_id: str
    kind: MemberKind
    artifact_ref: str
    content_hash: str
    schema_version: str = ARTIFACT_INVENTORY_MEMBER_V1


@dataclass(frozen=True, kw_only=True)
class ModelArtifactInventory:
    artifact_id: str
    role: ModelRole
    strategy_ids: tuple[str, ...]
    compatible_clock_ids: tuple[str, ...]
    target_contract_ref: str
    ordered_features: tuple[str, ...]
    members: tuple[ArtifactInventoryMember, ...]
    upstream_artifact_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    schema_version: str = MODEL_ARTIFACT_INVENTORY_V1


@dataclass(frozen=True, kw_only=True)
class ReleaseBinding:
    role: ModelRole
    strategy_id: str
    clock_id: str
    artifact_id: str
    ordered_features: tuple[str, ...]
    required_member_kinds: tuple[MemberKind, ...]
    schema_version: str = RELEASE_BINDING_V1

    @property
    def key(self) -> tuple[str, str, str]:
        return self.role, self.strategy_id, self.clock_id


@dataclass(frozen=True, kw_only=True)
class ReleaseRequirement:
    role: ModelRole
    strategy_id: str
    clock_id: str
    schema_version: str = RELEASE_REQUIREMENT_V1

    @property
    def key(self) -> tuple[str, str, str]:
        return self.role, self.strategy_id, self.clock_id


@dataclass(frozen=True, kw_only=True)
class ModelReleaseInventory:
    release_id: str
    deployment_id: str
    known_clock_ids: tuple[str, ...]
    artifacts: tuple[ModelArtifactInventory, ...]
    bindings: tuple[ReleaseBinding, ...]
    requirements: tuple[ReleaseRequirement, ...]
    artifact_manifest_ref: str
    evidence_refs: tuple[str, ...]
    promotion_receipt_ref: str | None = None
    schema_version: str = MODEL_RELEASE_V1


@dataclass(frozen=True, order=True)
class ReleaseIssue:
    path: str
    code: str
    detail: str


class ModelReleaseRefusal(ValueError):
    code = MODEL_RELEASE_REFUSAL

    def __init__(self, issues: tuple[ReleaseIssue, ...]) -> None:
        self.issues = issues
        detail = "; ".join(f"{item.code} at {item.path}" for item in issues)
        super().__init__(f"{self.code}: {detail}")


def _duplicates(values):
    return sorted({value for value in values if values.count(value) > 1})


def _add(issues, path, code, detail):
    issues.append(ReleaseIssue(path=path, code=code, detail=detail))


def _header_issues(release, issues):
    if not release.artifact_manifest_ref:
        _add(issues, "$.artifact_manifest_ref", "MISSING_ARTIFACT_MANIFEST", "empty")
    if not release.evidence_refs:
        _add(issues, "$.evidence_refs", "MISSING_EVIDENCE", "empty")
    for clock in _duplicates(release.known_clock_ids):
        _add(issues, "$.known_clock_ids", "DUPLICATE_CLOCK", clock)
    return set(release.known_clock_ids)


def _artifact_issues(release, issues):
    artifact_ids = tuple(item.artifact_id for item in release.artifacts)
    for artifact_id in _duplicates(artifact_ids):
        _add(issues, "$.artifacts", "DUPLICATE_ARTIFACT", artifact_id)
    artifacts = {item.artifact_id: item for item in release.artifacts}
    for index, artifact in enumerate(release.artifacts):
        path = f"$.artifacts[{index}]"
        if not artifact.ordered_features:
            _add(issues, f"{path}.ordered_features", "EMPTY_FEATURE_ORDER", "empty")
        for feature in _duplicates(artifact.ordered_features):
            _add(issues, f"{path}.ordered_features", "DUPLICATE_FEATURE", feature)
        member_ids = tuple(item.member_id for item in artifact.members)
        for member_id in _duplicates(member_ids):
            _add(issues, f"{path}.members", "DUPLICATE_MEMBER", member_id)
        for member_index, member in enumerate(artifact.members):
            member_path = f"{path}.members[{member_index}]"
            if not member.artifact_ref:
                _add(issues, f"{member_path}.artifact_ref", "MISSING_MEMBER_REF", "empty")
            if not member.content_hash.startswith("sha256:") or len(member.content_hash) != 71:
                _add(issues, f"{member_path}.content_hash", "INVALID_MEMBER_HASH", "sha256")
        if not any(item.kind == "estimator" for item in artifact.members):
            _add(issues, f"{path}.members", "MISSING_ESTIMATOR_MEMBER", artifact.artifact_id)
    return artifacts


def _binding_issues(release, issues, known_clocks, artifacts):
    binding_keys = tuple(item.key for item in release.bindings)
    for key in _duplicates(binding_keys):
        _add(issues, "$.bindings", "DUPLICATE_BINDING", repr(key))
    for index, binding in enumerate(release.bindings):
        path = f"$.bindings[{index}]"
        if binding.clock_id not in known_clocks:
            _add(issues, f"{path}.clock_id", "UNKNOWN_CLOCK", binding.clock_id)
        artifact = artifacts.get(binding.artifact_id)
        if artifact is None:
            _add(issues, f"{path}.artifact_id", "UNKNOWN_ARTIFACT", binding.artifact_id)
            continue
        if artifact.role != binding.role:
            _add(issues, f"{path}.role", "ROLE_MISMATCH", artifact.role)
        if binding.strategy_id not in artifact.strategy_ids and "*" not in artifact.strategy_ids:
            _add(issues, f"{path}.strategy_id", "STRATEGY_MISMATCH", binding.strategy_id)
        if binding.clock_id not in artifact.compatible_clock_ids:
            _add(issues, f"{path}.clock_id", "CLOCK_MISMATCH", binding.clock_id)
        if binding.ordered_features != artifact.ordered_features:
            _add(issues, f"{path}.ordered_features", "FEATURE_ORDER_MISMATCH", "order")
        kinds = {item.kind for item in artifact.members}
        for kind in binding.required_member_kinds:
            if kind not in kinds:
                _add(issues, f"{path}.required_member_kinds", f"MISSING_{kind.upper()}_MEMBER", binding.artifact_id)
    return binding_keys


def _requirement_issues(release, issues, binding_keys):
    requirement_keys = tuple(item.key for item in release.requirements)
    for key in sorted(set(requirement_keys)):
        count = binding_keys.count(key)
        if count == 0:
            _add(issues, "$.requirements", "MISSING_BINDING", repr(key))
        elif count > 1:
            _add(issues, "$.requirements", "AMBIGUOUS_BINDING", repr(key))
    for key in sorted(set(binding_keys) - set(requirement_keys)):
        _add(issues, "$.bindings", "UNDECLARED_BINDING", repr(key))
def release_issues(release: ModelReleaseInventory) -> tuple[ReleaseIssue, ...]:
    """Return deterministic defects without reading or loading artifacts."""
    issues: list[ReleaseIssue] = []
    known_clocks = _header_issues(release, issues)
    artifacts = _artifact_issues(release, issues)
    binding_keys = _binding_issues(release, issues, known_clocks, artifacts)
    _requirement_issues(release, issues, binding_keys)
    return tuple(sorted(issues))


def require_complete_release(release: ModelReleaseInventory) -> ModelReleaseInventory:
    issues = release_issues(release)
    if issues:
        raise ModelReleaseRefusal(issues)
    return release


__all__ = [
    "ARTIFACT_INVENTORY_MEMBER_V1", "MODEL_ARTIFACT_INVENTORY_V1",
    "MODEL_RELEASE_REFUSAL", "MODEL_RELEASE_V1", "RELEASE_BINDING_V1",
    "RELEASE_REQUIREMENT_V1", "ArtifactInventoryMember", "ModelArtifactInventory",
    "ModelReleaseInventory", "ModelReleaseRefusal", "ReleaseBinding", "ReleaseIssue",
    "ReleaseRequirement", "release_issues", "require_complete_release",
]
