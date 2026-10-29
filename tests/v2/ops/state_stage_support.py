"""Shared fixtures and helpers for the state-stage test modules."""
from __future__ import annotations

import json
from pathlib import Path

from checks import phase5_release_assembly as assembly
from engine.v2.models import (
    ArtifactInventoryMember,
    ArtifactMember,
    ModelArtifactInventory,
    ModelBinding,
    ModelRelease,
    ModelReleaseInventory,
    ReleaseBinding,
    ReleaseRequirement,
    deployment,
)
from engine.v2.models import release_layout as phase5_release
from engine.v2.ops.state_stage import StateStageParameters

REPO_ROOT = Path(__file__).resolve().parents[3]
UNRELATED_MEMBER_ID = "payoff_line:STR-THRU"


def fixture(release_id, deployment_id="d1", binding_id="b1"):
    payload = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": ["x"],
                          "outputs": [{"name": "prediction", "intercept": 1.0, "coefficients": [2.0]}]},
                         sort_keys=True).encode()
    member_hash = phase5_release.sha256_bytes(payload)
    member = ArtifactMember(name="estimator", path="unused.json", content_hash=member_hash)
    binding = ModelBinding(binding_id=binding_id, model_id="m1", role="size", strategy_id="*",
                           decision_clock_id="entry-close", adapter="json-linear.v1",
                           feature_order=("x",), output_names=("prediction",), members=(member,))
    release = ModelRelease(release_id=release_id, deployment_id=deployment_id, bindings=(binding,))
    inv_member = ArtifactInventoryMember(member_id="m1:estimator", kind="estimator",
                                         artifact_ref="artifact://m1", content_hash=member_hash)
    artifact = ModelArtifactInventory(artifact_id="m1", role="size", strategy_ids=("*",),
                                      compatible_clock_ids=("entry-close",),
                                      target_contract_ref="return.v1", ordered_features=("x",),
                                      members=(inv_member,))
    inv_binding = ReleaseBinding(role="size", strategy_id="*", clock_id="entry-close",
                                 artifact_id="m1", ordered_features=("x",),
                                 required_member_kinds=("estimator",))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id=deployment_id, known_clock_ids=("entry-close",),
        artifacts=(artifact,), bindings=(inv_binding,),
        requirements=(ReleaseRequirement(role="size", strategy_id="*", clock_id="entry-close"),),
        artifact_manifest_ref="manifest://r", evidence_refs=("evidence://r",))
    return release, inventory, {member_hash: payload}


def spec(member_id):
    return next(spec for spec in phase5_release.STATE_SPECS if spec.member_id == member_id)


def stage_prior(root, release_id="prior1", deployment_id="d1", binding_id="b1",
                prior_names=None, prior_sources=None):
    release, inventory, payloads = fixture(release_id, deployment_id=deployment_id,
                                           binding_id=binding_id)
    deployment.stage_release(phase5_release.deployment_root(root), release, inventory, payloads)
    rows = []
    for member_id in (*assembly.POOL_MEMBER_IDS, assembly.CHOOSER_MEMBER_ID, UNRELATED_MEMBER_ID):
        payload = f"prior-{member_id}".encode()
        content_hash, rel = phase5_release.write_object(root, payload)
        name = "prior.bin" if prior_names is None else prior_names[member_id]
        obj = {"name": name, "path": rel, "content_hash": content_hash, "bytes": len(payload)}
        rows.append(phase5_release.member_row(spec(member_id), phase5_release.STAGED, [obj], "prior"))
    sources = {"pool_cutoff": "2026-09-01"} if prior_sources is None else prior_sources
    body = phase5_release.manifest_body(release_id, deployment_id, rows, sources)
    phase5_release.write_manifest(root, body)
    return inventory, body


STATES = {
    "paired.json": ("paired_residual_pool", "mv|cr|2026-10-01"),
    "board.json": ("board_analog_matcher", "STR-THRU|0.5000|2026-10-01"),
    "trailing.json": ("trailing_pnl_cutoff", "hist|2026-10-01"),
}


def trailing_payload(object_name: str) -> bytes:
    """A valid trailing-cutoff artifact JSON whose lineage ends at the object name's month."""
    from engine.v2.models.lineage import DataDependency, Lineage
    from engine.v2.models.trailing_cutoff_artifact import (
        PNL_HISTORY_ID, make_trailing_cutoff_artifact)
    month = object_name.rsplit("|", 1)[-1][:7] + "-01"
    artifact = make_trailing_cutoff_artifact(
        month=month, cutoff=50.0,
        lineage=Lineage(data=(DataDependency(table=PNL_HISTORY_ID, end_exclusive=month),)))
    return json.dumps(artifact.payload()).encode()


def state_bytes(member: str, object_name: str, file_name: str) -> bytes:
    """Fixture payload for a staged state file: valid trailing JSON, else opaque bytes."""
    if member == "trailing_pnl_cutoff":
        try:
            return trailing_payload(object_name)
        except ValueError:
            return f"state-{file_name}".encode()
    return f"state-{file_name}".encode()


BOOTSTRAP = {
    "tier4_dir": "/tmp/tier4",
    "panel_catalog": "/tmp/panel",
    "panel_snapshot_id": "snap",
    "panel_artifact_root": "/tmp/artifact",
}


def params(**changes):
    values = dict(expected_ids=("phase5_state_stage",), release_root="/r", prior_release_id="prior1",
                  release_id="cand1", cutoff="2026-10-01", state_files=tuple(sorted(STATES)),
                  input_bindings={**{name: f"job_x#training/{name}" for name in STATES},
                                  "legacy_manifest.json": "artifact_manifest"})
    if changes.get("mode") == "bootstrap" and "bootstrap" not in changes:
        values["bootstrap"] = dict(BOOTSTRAP)
    values.update(changes)
    return StateStageParameters(**values)
