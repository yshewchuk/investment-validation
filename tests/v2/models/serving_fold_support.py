"""Public synthetic artifact builders shared by authoring and loader tests."""
import hashlib
import io
import json

import joblib

from engine.v2.models import release_layout as layout
from engine.v2.foundation import content_hash, from_document, to_document
from engine.v2.models.serving_fold_preflight import describe_size_fold
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
from engine.v2.models.serving_folds import ChooserFoldPolicy, SizeFoldPolicy
from tools import phase5_prepare_release as prep

OUTPUTS = {"implied_t1": ("driver_prediction", "pred_im_t1_d14"),
           "runup_move": ("pred_runup_abs_move_d14", "pred_runup_abs_move_d14")}


def fold_bytes(cache, **dump_options):
    stream = io.BytesIO()
    joblib.dump(cache, stream, **dump_options)
    return stream.getvalue()


def fold_name(cache):
    month = cache["fold_start"].replace("-", "")[:6]
    return f"{cache['model_id']}_{month}_{cache['tier3_snapshot'][:12]}.joblib"


def staged_fold_object(cache, manifest, dep_root):
    """Write one synthetic size fold into ``dep_root`` and describe its object."""
    raw = fold_bytes(cache)
    policy = SizeFoldPolicy(
        model_id=cache["model_id"], feature_order=tuple(cache["features"]),
        panel_sha256=cache["tier3_snapshot"], interval_floor=0.0)
    descriptor = describe_size_fold(raw, fold_name(cache), policy, manifest)
    path = dep_root / descriptor.estimator.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": descriptor.estimator.path,
        "content_hash": descriptor.estimator.content_hash,
        "name": fold_name(cache), "bytes": len(raw),
        "serving_fold": to_document(descriptor),
    }


def write_catalog(root, release_id, fold_object=None):
    """Write the required catalog, optionally with one synthetic size fold."""
    members = ([] if fold_object is None else [{
        "member_id": "tier4_folds:size", "status": "STAGED", "objects": [fold_object],
    }])
    body = {"schema_version": "phase5_staged_release.v1.0", "release_id": release_id,
            "deployment_id": "d1", "members": members, "sources": {}}
    body["manifest_hash"] = content_hash(
        {k: v for k, v in body.items() if k != "manifest_hash"})
    root.mkdir(parents=True, exist_ok=True)
    (root / "phase5_release.json").write_text(json.dumps(body, sort_keys=True))


def chooser_policy(role, cache, *, floor=0.0):
    return ChooserFoldPolicy(role=role, model_id=cache["model_id"],
                             feature_order=cache["features"], panel_sha256=cache["tier3_snapshot"],
                             interval_floor=floor)


def release_inputs(role, cache, *, output=None, clock="entry-close", release_id="r1"):
    features, model_id = tuple(cache["features"]), cache["model_id"]
    output = ({"size": "forecast_abs_move", **{key: value[0] for key, value in OUTPUTS.items()}}[role]
              if output is None else output)
    outputs = (output,) if isinstance(output, str) else tuple(output)
    raw = json.dumps({"schema_version": "linear_estimator.v1.0", "feature_order": features,
                      "outputs": [{"name": name, "intercept": 0,
                                   "coefficients": [0] * len(features)} for name in outputs]}).encode()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    binding = ModelBinding(binding_id=f"{role}:*", model_id=model_id, role=role, strategy_id="*",
                           decision_clock_id=clock, adapter="json-linear.v1",
                           feature_order=features, output_names=outputs, members=(ArtifactMember(
                               name="estimator", path="source", content_hash=digest),))
    inventory = ModelReleaseInventory(
        release_id=release_id, deployment_id="d1", known_clock_ids=(clock,),
        artifacts=(ModelArtifactInventory(
            artifact_id=model_id, role=role, strategy_ids=("*",), compatible_clock_ids=(clock,),
            target_contract_ref="synthetic.v1", ordered_features=features,
            members=(ArtifactInventoryMember(member_id=f"{role}:estimator", kind="estimator",
                                              artifact_ref="source", content_hash=digest),)),),
        bindings=(ReleaseBinding(role=role, strategy_id="*", clock_id=clock, artifact_id=model_id,
                                 ordered_features=features, required_member_kinds=("estimator",)),),
        requirements=(ReleaseRequirement(role=role, strategy_id="*", clock_id=clock),),
        artifact_manifest_ref="synthetic", evidence_refs=("synthetic",))
    return ModelRelease(release_id=release_id, deployment_id="d1", bindings=(binding,)), inventory, {digest: raw}


def fold_states(role, cache, *, explicit=True, data=None):
    found = {f"tier4_folds:{role}": {fold_name(cache): fold_bytes(cache) if data is None else data}}
    if role == "size":
        policy = SizeFoldPolicy(model_id=cache["model_id"], feature_order=cache["features"],
                                panel_sha256=cache["tier3_snapshot"], interval_floor=0.0)
        return prep.build_states(found, size_fold_policy=policy if explicit else None)
    return prep.build_states(found,
                             chooser_fold_policies={role: chooser_policy(role, cache)} if explicit else None)


def catalog_object(root, role, *, name=None):
    body = layout.read_manifest(root)
    row = next(row for row in body["members"] if row["member_id"] == f"tier4_folds:{role}")
    assert row["status"] == "STAGED"
    objects = row["objects"] if name is None else [obj for obj in row["objects"] if obj["name"] == name]
    assert len(objects) == 1
    return body, objects[0]


def published_manifest(root, release_id="r1"):
    # Read our synthetic artifact through the public typed-document boundary.
    return from_document(deployment.StagedManifest, json.loads(
        (root / "releases" / release_id / "manifest.json").read_text()))
