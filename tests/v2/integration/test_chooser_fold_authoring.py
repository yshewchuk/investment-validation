"""Offline chooser fold declarations using only synthetic temporary artifacts."""
# packages: engine.v2.models, engine.v2.data, engine.v2.foundation
import dataclasses
import hashlib
import io
import json
from types import SimpleNamespace

import joblib
import pytest

from checks import phase5_release as layout
from engine.v2.foundation import DocumentError, content_hash, from_document, to_document
from engine.v2.models import (
    ArtifactInventoryMember, ArtifactMember, ModelArtifactInventory, ModelBinding,
    ModelRelease, ModelReleaseInventory, ReleaseBinding, ReleaseRequirement, deployment,
)
from engine.v2.models.serving_folds import (
    ChooserFoldPolicy, ChooserServingFoldDescriptor, ServingFoldDescriptor, SizeFoldPolicy,
)
from tests.test_phase5_pinned_panel import ALL_ONE, build_pinned
from tools import phase5_prepare_release as prep
from tools import phase5_serving_folds as folds

OUTPUTS = {"implied_t1": ("driver_prediction", "pred_im_t1_d14"),
           "runup_move": ("pred_runup_abs_move_d14", "pred_runup_abs_move_d14")}


class NoInference:
    def predict(self, rows):
        raise AssertionError("authoring must not predict")

    def fit(self, *args):
        raise AssertionError("authoring must not fit")


def _cache(role, panel="a" * 64):
    return {"model_id": f"synthetic-{role}", "features": ["x", "y"],
            "tier3_snapshot": panel, "fold_start": "2026-02-01", "estimator": NoInference(),
            "pool_pred": [1.0, 3.0], "pool_res": [-0.5, 0.25]}


def _bytes(cache):
    stream = io.BytesIO()
    joblib.dump(cache, stream)
    return stream.getvalue()


def _name(cache):
    month = cache["fold_start"].replace("-", "")[:6]
    return f"{cache['model_id']}_{month}_{cache['tier3_snapshot'][:12]}.joblib"


def _policy(role, cache, *, floor=0.0):
    return ChooserFoldPolicy(role=role, model_id=cache["model_id"],
                             feature_order=cache["features"], panel_sha256=cache["tier3_snapshot"],
                             interval_floor=floor)


def _inputs(role, cache, *, output=None, clock="entry-close", release_id="r1"):
    features, model_id = tuple(cache["features"]), cache["model_id"]
    output = OUTPUTS[role][0] if output is None else output
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


def _retry_inputs(role, cache):
    parts = [_inputs(role, cache), _inputs("auxiliary", _cache("auxiliary"), output=("left", "right"))]
    bindings = tuple(dataclasses.replace(part[0].bindings[0], members=(
        part[0].bindings[0].members[0], dataclasses.replace(part[0].bindings[0].members[0], name="metadata")))
        for part in parts)
    release = dataclasses.replace(parts[0][0], bindings=bindings)
    inventory = dataclasses.replace(parts[0][1], **{
        field: tuple(getattr(part[1], field)[0] for part in parts)
        for field in ("artifacts", "bindings", "requirements")})
    return release, inventory, {key: value for part in parts for key, value in part[2].items()}


def _states(role, cache, *, explicit=True):
    return prep.build_states({f"tier4_folds:{role}": {_name(cache): _bytes(cache)}},
                             chooser_fold_policies={role: _policy(role, cache)} if explicit else None)


def _object(root, role, *, name=None):
    body = layout.read_manifest(root)
    row = next(row for row in body["members"] if row["member_id"] == f"tier4_folds:{role}")
    assert row["status"] == "STAGED"
    objects = row["objects"] if name is None else [obj for obj in row["objects"] if obj["name"] == name]
    assert len(objects) == 1
    return body, objects[0]


def _assert_published_descriptor(root, role, cache, *, release_id="r1", clock="entry-close", floor=0.0, count=2):
    body, obj = _object(root, role, name=_name(cache))
    digest = hashlib.sha256(_bytes(cache)).hexdigest()
    parent = deployment._read_manifest(root / "deployment", release_id)
    expected = {
        "schema_version": "chooser_serving_fold_descriptor.v1.0",
        "parent_release_id": release_id, "parent_release_hash": parent.release_hash,
        "policy": {"schema_version": "chooser_fold_policy.v1.0", "role": role,
                   "model_id": cache["model_id"], "feature_order": cache["features"],
                   "panel_sha256": cache["tier3_snapshot"], "interval_floor": floor,
                   "interval_policy": "chooser-heldout-deciles.v1"},
        "fold_start": cache["fold_start"], "decision_clock_id": clock, "pool_count": count,
        "pool_pred_field": "pool_pred", "pool_res_field": "pool_res",
        "output_name": OUTPUTS[role][0], "producer_output": OUTPUTS[role][1],
        "estimator": {"schema_version": "artifact_member.v1.0", "name": "estimator",
                      "path": f"objects/{digest}", "content_hash": f"sha256:{digest}"},
    }
    assert obj["serving_fold"] == expected
    descriptor = from_document(ChooserServingFoldDescriptor, obj["serving_fold"])
    assert descriptor.descriptor_hash == content_hash(expected)
    assert obj["path"] == f"objects/{digest}"
    assert obj["content_hash"] == f"sha256:{digest}"
    assert obj["name"] == _name(cache)
    assert obj["bytes"] == len(_bytes(cache))
    return body, obj


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("count,month,clock,features,floor,release_id", [
    (2, "2026-02-01", "entry-close", ["x", "y"], 0.0, "r1"),
    (0, "2026-05-01", "next-open", ["b", "c", "a"], None, "empty-release"),
    (1, "2025-11-01", "next-open", ["y", "x"], 1.25, "thin-release"),
])
def test_real_authoring_preserves_exact_role_identity_and_metadata(
        tmp_path, role, count, month, clock, features, floor, release_id):
    cache = _cache(role)
    cache.update(model_id=f"{release_id}-{role}", fold_start=month, features=features,
                 pool_pred=[1.0] * count, pool_res=[-0.5] * count)
    args = _inputs(role, cache, clock=clock, release_id=release_id)
    states = prep.build_states({f"tier4_folds:{role}": {_name(cache): _bytes(cache)}},
                               chooser_fold_policies={role: _policy(role, cache, floor=floor)})
    prep.write_release(tmp_path, *args, states)
    body, obj = _assert_published_descriptor(tmp_path, role, cache, release_id=release_id,
                                            clock=clock, floor=floor, count=count)
    descriptor = from_document(ChooserServingFoldDescriptor, obj["serving_fold"])
    assert to_document(descriptor) == obj["serving_fold"]
    with pytest.raises(DocumentError) as error:
        from_document(ServingFoldDescriptor, obj["serving_fold"])
    assert error.value.code == "UNSUPPORTED_VERSION"
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.policy.role = "size"
    with pytest.raises(dataclasses.FrozenInstanceError):
        descriptor.parent_release_id = "mutated"
    changed = dataclasses.replace(descriptor, policy=dataclasses.replace(
        descriptor.policy, interval_floor=2.5))
    assert changed.descriptor_hash != descriptor.descriptor_hash
    unsigned = {key: value for key, value in body.items() if key != "manifest_hash"}
    assert content_hash(unsigned) == body["manifest_hash"]
    unsigned["members"] = json.loads(json.dumps(unsigned["members"]))
    selected = next(row for row in unsigned["members"] if row["member_id"] == f"tier4_folds:{role}")
    selected["objects"][0]["serving_fold"]["pool_count"] = 3
    assert content_hash(unsigned) != body["manifest_hash"]
    prep.write_release(tmp_path, *args, states)
    assert layout.read_manifest(tmp_path) == body
    assert deployment.current_pointer(tmp_path / "deployment") is None


@pytest.mark.parametrize("identity", ["parent_release_id", "parent_release_hash", "role", "estimator"])
def test_descriptor_hash_binds_parent_role_and_estimator_identity(tmp_path, identity):
    role, cache = "implied_t1", _cache("implied_t1")
    prep.write_release(tmp_path, *_inputs(role, cache), _states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    expected = json.loads(json.dumps(obj["serving_fold"]))
    original = from_document(ChooserServingFoldDescriptor, expected)
    if identity == "role":
        expected["policy"]["role"] = "runup_move"
        changed = dataclasses.replace(original, policy=dataclasses.replace(original.policy, role="runup_move"))
    elif identity == "estimator":
        expected["estimator"]["content_hash"] = "sha256:" + "b" * 64
        changed = dataclasses.replace(original, estimator=dataclasses.replace(
            original.estimator, content_hash="sha256:" + "b" * 64))
    else:
        value = "other-release" if identity == "parent_release_id" else "sha256:" + "c" * 64
        expected[identity] = value
        changed = dataclasses.replace(original, **{identity: value})
    assert changed.descriptor_hash == content_hash(expected)
    assert changed.descriptor_hash != original.descriptor_hash


@pytest.mark.parametrize("role", OUTPUTS)
def test_policy_absence_keeps_opaque_catalog_compatibility(tmp_path, role):
    cache = _cache(role)
    prep.write_release(tmp_path, *_inputs(role, cache), _states(role, cache, explicit=False))
    _body, obj = _object(tmp_path, role)
    assert "serving_fold" not in obj


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("pinned_mode", [False, True])
@pytest.mark.parametrize("bad_later_month", [False, True])
def test_every_month_is_preflighted_and_published_before_catalog(
        tmp_path, role, pinned_mode, bad_later_month):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, data = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(data).hexdigest()
    first = _cache(role, panel_hash)
    later = _cache(role, panel_hash)
    later.update(fold_start="2026-03-01", pool_pred=[4.5],
                 pool_res=[float("nan") if bad_later_month else -1.25])
    # Insert later first so successful ordering cannot rely on mapping insertion order.
    payloads = {_name(later): _bytes(later), _name(first): _bytes(first)}
    states = prep.build_states({f"tier4_folds:{role}": payloads},
                               chooser_fold_policies={role: _policy(role, first)})
    out = tmp_path / "out"
    args = _inputs(role, first)
    if bad_later_month:
        with pytest.raises(prep.PrepareRefused) as error:
            prep.write_release(out, *args, states, pinned_panel=pinned)
        assert type(error.value) is prep.PrepareRefused
        assert str(error.value) == "chooser fold refused: invalid serving fold held-out pool"
        assert type(error.value.__cause__) is ValueError
        assert str(error.value.__cause__) == "invalid serving fold held-out pool"
        assert not out.exists()
        assert deployment.current_pointer(out / "deployment") is None
        return
    prep.write_release(out, *args, states, pinned_panel=pinned)
    body = layout.read_manifest(out)
    row = next(row for row in body["members"] if row["member_id"] == f"tier4_folds:{role}")
    assert [obj["name"] for obj in row["objects"]] == [_name(first), _name(later)]
    first_obj = _assert_published_descriptor(out, role, first, count=2)[1]
    later_obj = _assert_published_descriptor(out, role, later, count=1)[1]
    assert first_obj["content_hash"] != later_obj["content_hash"]
    assert content_hash(first_obj["serving_fold"]) != content_hash(later_obj["serving_fold"])
    assert content_hash({key: value for key, value in body.items() if key != "manifest_hash"}) == body["manifest_hash"]
    if pinned_mode:
        assert body["sources"]["panel_copy_sha256"] == panel_hash
    prep.write_release(out, *args, states, pinned_panel=pinned)
    assert layout.read_manifest(out) == body
    assert deployment.current_pointer(out / "deployment") is None


def test_size_and_both_chooser_roles_coexist_without_relabeling(tmp_path):
    caches = {role: _cache(role) for role in OUTPUTS}
    parts = [_inputs(role, cache) for role, cache in caches.items()]
    size = _cache("size")
    parts.append(_inputs("size", size, output="forecast_abs_move"))
    release = dataclasses.replace(parts[0][0], bindings=tuple(p[0].bindings[0] for p in parts))
    inventory = dataclasses.replace(parts[0][1], **{
        field: tuple(getattr(p[1], field)[0] for p in parts)
        for field in ("artifacts", "bindings", "requirements")})
    payloads = {key: value for p in parts for key, value in p[2].items()}
    states = prep.build_states(
        {"tier4_folds:size": {_name(size): _bytes(size)},
         **{f"tier4_folds:{role}": {_name(cache): _bytes(cache)} for role, cache in caches.items()}},
        size_fold_policy=SizeFoldPolicy(model_id=size["model_id"], feature_order=("x", "y"),
                                       panel_sha256="a" * 64, interval_floor=0.0),
        chooser_fold_policies={role: _policy(role, cache) for role, cache in caches.items()})
    prep.write_release(tmp_path, release, inventory, payloads, states)
    descriptors = [_assert_published_descriptor(tmp_path, role, caches[role])[1]["serving_fold"]
                   for role in OUTPUTS]
    assert [d["policy"]["role"] for d in descriptors] == ["implied_t1", "runup_move"]
    assert [d["producer_output"] for d in descriptors] == ["pred_im_t1_d14", "pred_runup_abs_move_d14"]
    assert len({d["estimator"]["content_hash"] for d in descriptors}) == 2
    size_descriptor = _object(tmp_path, "size")[1]["serving_fold"]
    assert size_descriptor["schema_version"] == "serving_fold_descriptor.v1.0"
    assert size_descriptor["role"] == "size"
    assert size_descriptor["output_name"] == "forecast_abs_move"
    assert {d["schema_version"] for d in descriptors} == {"chooser_serving_fold_descriptor.v1.0"}
    assert len({d["parent_release_hash"] for d in [size_descriptor, *descriptors]}) == 1


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("pinned_mode", [False, True])
@pytest.mark.parametrize("reorder", ["bindings", "members", "both"])
def test_semantically_identical_reordered_retries_keep_original_manifest_and_catalog(
        tmp_path, role, pinned_mode, reorder):
    pinned, panel_hash = None, "a" * 64
    if pinned_mode:
        source = tmp_path / "pinned"
        source.mkdir()
        pinned, _store, _snapshot, _records, data = build_pinned(source, ALL_ONE)
        panel_hash = hashlib.sha256(data).hexdigest()
    cache = _cache(role, panel_hash)
    release, inventory, payloads = _retry_inputs(role, cache)
    states, out = _states(role, cache), tmp_path / "out"
    prep.write_release(out, release, inventory, payloads, states, pinned_panel=pinned)
    catalog, _obj = _assert_published_descriptor(out, role, cache)
    manifest = deployment._read_manifest(out / "deployment", "r1")
    bindings = release.bindings
    if reorder in ("members", "both"):
        bindings = tuple(dataclasses.replace(binding, members=tuple(reversed(binding.members)))
                         for binding in bindings)
    if reorder in ("bindings", "both"):
        bindings = tuple(reversed(bindings))
    reordered = dataclasses.replace(release, bindings=bindings)
    assert reordered != release
    prep.write_release(out, reordered, inventory, payloads, states, pinned_panel=pinned)
    assert deployment._read_manifest(out / "deployment", "r1") == manifest
    assert _assert_published_descriptor(out, role, cache)[0] == catalog
    assert deployment.current_pointer(out / "deployment") is None


@pytest.mark.parametrize("role,expected", [
    ("implied_t1", "sha256:d046274a8fe00a5ab5b167671fb592ec510d29083b4ef555fd622a38ee02b8a0"),
    ("runup_move", "sha256:9e9f90638fe05f02f1a57569a79acba6380a72e7f9c3e32f2a71f3370d1f6adb"),
])
def test_public_semantic_release_hash_preserves_preexisting_identity(role, expected):
    # Goldens captured from the existing calculation before exposing its public name.
    release, _inventory, _payloads = _retry_inputs(role, _cache(role))
    assert "semantic_release_hash" in deployment.__all__
    assert deployment.semantic_release_hash(release) == expected
    assert deployment._release_hash(release) == expected
    reordered = dataclasses.replace(release, bindings=tuple(dataclasses.replace(
        binding, members=tuple(dataclasses.replace(member, path=f"relocated/{member.name}")
                               for member in reversed(binding.members)))
        for binding in reversed(release.bindings)))
    assert deployment.semantic_release_hash(reordered) == expected


@pytest.mark.parametrize("role", OUTPUTS)
def test_descriptor_parent_calls_public_release_identity(tmp_path, role, monkeypatch):
    cache = _cache(role)
    prep.write_release(tmp_path, *_retry_inputs(role, cache), _states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    manifest = deployment._read_manifest(tmp_path / "deployment", "r1")
    policy = _policy(role, cache)
    validated = folds.preflight_chooser_fold(_bytes(cache), _name(cache), policy, manifest.release)
    calls = []

    def public_hash(release):
        calls.append(release)
        return deployment.semantic_release_hash(release)

    def private_call(*args):
        pytest.fail("descriptor authoring must use the public release identity")

    monkeypatch.setattr(deployment, "_release_hash", private_call)
    monkeypatch.setattr(folds, "semantic_release_hash", public_hash)
    descriptor = folds.descriptor_from_preflight(validated, policy, manifest)
    assert calls == [manifest.release, validated.release]
    assert to_document(descriptor) == obj["serving_fold"]


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("legacy_hash", [False, True])
def test_parent_storage_paths_do_not_change_semantics_or_manifest_hash_ownership(tmp_path, role, legacy_hash):
    cache = _cache(role)
    prep.write_release(tmp_path, *_retry_inputs(role, cache), _states(role, cache))
    _body, obj = _assert_published_descriptor(tmp_path, role, cache)
    manifest = deployment._read_manifest(tmp_path / "deployment", "r1")
    if legacy_hash:
        manifest = dataclasses.replace(manifest,
            release_hash=deployment._legacy_release_hash(manifest.release),
            release_hash_version=deployment.RELEASE_HASH_MEMBER_V1)
    assert deployment._manifest_hash_matches(manifest)
    relocated = dataclasses.replace(manifest.release, bindings=tuple(dataclasses.replace(
        binding, members=tuple(dataclasses.replace(member, path=f"relocated/{member.name}")
                               for member in binding.members)) for binding in manifest.release.bindings))
    assert relocated != manifest.release
    policy = _policy(role, cache)
    validated = folds.preflight_chooser_fold(_bytes(cache), _name(cache), policy, relocated)
    descriptor = folds.descriptor_from_preflight(validated, policy, manifest)
    expected = {**obj["serving_fold"], "parent_release_hash": manifest.release_hash}
    assert to_document(descriptor) == expected
    assert descriptor.descriptor_hash == content_hash(expected)


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("identity", ["release_id", "deployment_id", "model_id", "adapter", "clock",
                                     "feature_order", "output_order", "member_hash", "member_name"])
def test_retained_parent_guard_rejects_actual_identity_changes(tmp_path, role, identity):
    cache, policy = _cache(role), _policy(role, _cache(role))
    prep.write_release(tmp_path, *_retry_inputs(role, cache), _states(role, cache))
    manifest = deployment._read_manifest(tmp_path / "deployment", "r1")
    validated = folds.preflight_chooser_fold(_bytes(cache), _name(cache), policy, manifest.release)
    release, auxiliary = manifest.release, manifest.release.bindings[1]
    if identity in ("release_id", "deployment_id"):
        release = dataclasses.replace(release, **{identity: "different"})
    else:
        changes = {"model_id": {"model_id": "different"}, "adapter": {"adapter": "different"},
                   "clock": {"decision_clock_id": "different"},
                   "feature_order": {"feature_order": tuple(reversed(auxiliary.feature_order))},
                   "output_order": {"output_names": tuple(reversed(auxiliary.output_names))}}
        if identity in ("member_hash", "member_name"):
            field, value = ("content_hash", "sha256:" + "b" * 64) if identity == "member_hash" else ("name", "different")
            changed = dataclasses.replace(auxiliary.members[0], **{field: value})
            changes[identity] = {"members": (changed, *auxiliary.members[1:])}
        release = dataclasses.replace(release, bindings=(release.bindings[0],
                                                        dataclasses.replace(auxiliary, **changes[identity])))
    # Keep the asserted hash unchanged: compare actual releases, not just caller-supplied hash strings.
    substituted = dataclasses.replace(manifest, release=release)
    with pytest.raises(ValueError) as error:
        folds.descriptor_from_preflight(validated, policy, substituted)
    assert type(error.value) is ValueError
    assert str(error.value) == "staged release disagrees with validated fold"
    assert error.value.__cause__ is None
    assert deployment._read_manifest(tmp_path / "deployment", "r1") == manifest


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("count", [0, 1])
def test_empty_and_thin_pools_keep_explicit_counts(role, count):
    cache = _cache(role)
    cache.update(pool_pred=[1.0] * count, pool_res=[-0.5] * count)
    data, policy, release = _bytes(cache), _policy(role, cache), _inputs(role, cache)[0]
    validated = folds.preflight_chooser_fold(data, _name(cache), policy, release)
    expected = {"fold_start": "2026-02-01", "decision_clock_id": "entry-close",
                "digest": hashlib.sha256(data).hexdigest(), "pool_count": count,
                "policy": policy, "release": release}
    assert vars(validated) == expected
    assert {field.name for field in dataclasses.fields(validated)} == set(expected)
    with pytest.raises(dataclasses.FrozenInstanceError):
        validated.pool_count = 100


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("defect", ["role", "model", "features", "clock-release", "output",
                                    "duplicate", "policy", "floor", "schema", "panel",
                                    "month", "name", "estimator", "absent", "shape", "length", "nan"])
def test_bad_inputs_refuse_before_output_creation(tmp_path, role, defect):
    cache = _cache(role)
    release, inv, payloads = _inputs(role, cache)
    policy, name = _policy(role, cache), _name(cache)
    if defect in ("role", "model", "features", "output", "clock-release", "duplicate"):
        binding = release.bindings[0]
        changes = {"role": {"role": "size"}, "model": {"model_id": "wrong"},
                   "features": {"feature_order": ("y", "x")},
                   "output": {"output_names": ("runup_move_prediction",)},
                   "clock-release": {"decision_clock_id": "wrong"}, "duplicate": {}}
        binding = dataclasses.replace(binding, **changes[defect])
        release = dataclasses.replace(release, bindings=(binding,) * (2 if defect == "duplicate" else 1))
    elif defect in ("policy", "floor", "schema"):
        policy = dataclasses.replace(policy, **{"policy": {"interval_policy": "size-heldout-deciles.v1"},
                                                "floor": {"interval_floor": float("nan")},
                                                "schema": {"schema_version": "wrong"}}[defect])
    elif defect == "panel":
        cache["tier3_snapshot"] = "a" * 12 + "b" * 52
    elif defect == "month":
        cache["fold_start"] = "2026-02-02"
    elif defect == "name":
        name = "wrong.joblib"
    elif defect in ("estimator", "absent"):
        cache.pop("estimator" if defect == "estimator" else "pool_res")
    else:
        cache["pool_res"] = {"shape": [[1.0, 2.0]], "length": [1.0],
                             "nan": [float("nan"), 1.0]}[defect]
    states = prep.build_states({f"tier4_folds:{role}": {name: _bytes(cache)}},
                               chooser_fold_policies={role: policy})
    out = tmp_path / "out"
    expected_type = deployment.StagingRefused if defect == "clock-release" else prep.PrepareRefused
    with pytest.raises(expected_type) as error:
        prep.write_release(out, release, inv, payloads, states)
    assert type(error.value) is expected_type
    if defect == "clock-release":
        assert error.value.code == "MODEL_DEPLOYMENT_REFUSED"
        assert {(issue.path, issue.code) for issue in error.value.issues} == {
            ("$.bindings", "MISSING_INFERENCE_BINDING"),
            (f"$.bindings[{role}:*]", "UNBOUND_IN_INVENTORY"),
        }
    else:
        if defect in ("role", "model", "features", "output", "duplicate"):
            diagnostic = f"{role} serving policy disagrees with release binding"
        elif defect in ("policy", "floor", "schema"):
            diagnostic = "invalid chooser serving policy"
        elif defect in ("panel", "estimator"):
            diagnostic = "serving fold header disagrees with policy"
        elif defect == "month":
            diagnostic = "invalid serving fold month"
        elif defect == "name":
            diagnostic = "serving fold filename disagrees with header"
        else:
            diagnostic = "invalid serving fold held-out pool"
        assert str(error.value) == f"chooser fold refused: {diagnostic}"
        assert type(error.value.__cause__) is ValueError
        assert str(error.value.__cause__) == diagnostic
    if defect != "clock-release":  # Deployment, not the cache header, owns clock-inventory compatibility.
        assert not out.exists()
    assert not (out / "phase5_release.json").exists()
    assert deployment.current_pointer(out / "deployment") is None


@pytest.mark.parametrize("role", OUTPUTS)
def test_bound_precedes_decode_and_each_authoring_decodes_once(tmp_path, role, monkeypatch):
    cache = _cache(role)
    data, name, policy = _bytes(cache), _name(cache), _policy(role, cache)
    calls, real = [], joblib.load
    monkeypatch.setattr(joblib, "load", lambda *a, **k: calls.append(1) or real(*a, **k))
    monkeypatch.setattr(folds, "MAX_FOLD_BYTES", len(data) - 1)
    with pytest.raises(ValueError) as error:
        folds.preflight_chooser_fold(data, name, policy, _inputs(role, cache)[0])
    assert type(error.value) is ValueError
    assert str(error.value) == "serving fold exceeds byte limit"
    assert error.value.__cause__ is None
    assert calls == []
    monkeypatch.setattr(folds, "MAX_FOLD_BYTES", len(data))
    prep.write_release(tmp_path, *_inputs(role, cache), _states(role, cache))
    assert calls == [1]
    prep.write_release(tmp_path, *_inputs(role, cache), _states(role, cache))
    assert calls == [1, 1]


@pytest.mark.parametrize("role", OUTPUTS)
def test_pinned_authoring_requires_full_panel_hash_and_retained_destination(tmp_path, role, monkeypatch):
    (tmp_path / "pinned").mkdir()
    pinned, _store, _snapshot, _records, data = build_pinned(tmp_path / "pinned", ALL_ONE)
    panel_hash = hashlib.sha256(data).hexdigest()
    cache = _cache(role, panel_hash)
    out = tmp_path / "good"
    prep.write_release(out, *_inputs(role, cache), _states(role, cache), pinned_panel=pinned)
    body, obj = _object(out, role)
    assert body["sources"]["panel_copy_sha256"] == panel_hash
    assert obj["serving_fold"]["policy"]["panel_sha256"] == panel_hash
    bad = _cache(role, panel_hash[:12] + "0" * 52)
    refused = tmp_path / "bad-panel"
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(refused, *_inputs(role, bad), _states(role, bad), pinned_panel=pinned)
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold policy does not match the pinned panel"
    assert error.value.__cause__ is None
    assert not refused.exists()
    real = prep.write_object

    def wrong_destination(root, payload):
        digest, rel = real(root, payload)
        (layout.deployment_root(root) / rel).write_bytes(b"substituted")
        return digest, rel

    monkeypatch.setattr(prep, "write_object", wrong_destination)
    refused = tmp_path / "bad-destination"
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(refused, *_inputs(role, cache), _states(role, cache), pinned_panel=pinned)
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold object bytes differ from preflight"
    assert error.value.__cause__ is None
    assert not (refused / "phase5_release.json").exists()
    assert deployment.current_pointer(refused / "deployment") is None


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("failure,features,floor", [
    (None, ("x", "y"), 0.0), (None, ("y", "x"), 1.25),
    (None, ("b", "c", "a"), None), ("output", ("b", "c", "a"), 1.25),
    ("recipe", ("b", "c", "a"), None),
])
def test_real_cli_obtains_role_policy_without_fitting_or_data_reads(
        tmp_path, role, monkeypatch, capsys, failure, features, floor):
    from engine.data.features import tier4
    from engine.v2.models import inventory

    cache = _cache(role)
    cache["features"] = list(features)
    release, inv, payloads = _inputs(role, cache)
    monkeypatch.setattr(inventory, "current_release_inventory", lambda: (inv, ()))
    monkeypatch.setattr(prep, "model_release", lambda *a, **k: (release, payloads))
    monkeypatch.setattr(prep, "modules_available", lambda *a: (False, "unused"))
    monkeypatch.setattr(tier4, "size_feature_model", lambda: SimpleNamespace(
        model_id="unused-size", features=("size-x",), interval_floor=0.0, produces="pred_abs_move"))
    calls = []
    recipe_error = tier4.Tier4Error("private recipe feature or target detail")

    def producer():
        calls.append(role)
        if failure == "recipe":
            raise recipe_error
        return SimpleNamespace(model_id=cache["model_id"], features=features,
                               interval_floor=floor, produces="scaled" if failure == "output" else OUTPUTS[role][1])

    def never(*args, **kwargs):
        pytest.fail("unrelated producer, fitting or live data read")

    for candidate, method in (("implied_t1", "im_t1_feature_model"), ("runup_move", "runup_move_feature_model")):
        monkeypatch.setattr(tier4, method, producer if role == candidate else never)
    monkeypatch.setattr(tier4, "serving_model", never)
    monkeypatch.setattr(tier4.registry_mod, "load_registry", never)
    monkeypatch.setattr(tier4.store, "read_table", never)
    monkeypatch.setattr(tier4.store, "file_sha256", never)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / _name(cache)).write_bytes(_bytes(cache))
    out = tmp_path / "out"
    argv = ["--release-id", "r1", "--out", str(out), "--tier4-dir", str(cache_dir),
            "--tier3-snapshot", "a" * 64]
    if failure == "recipe":
        args = SimpleNamespace(release_id="r1", tier4_dir=cache_dir, tier3_snapshot="a" * 64,
                               fold_month=None, training_root=None, frozen_state=None)
        with pytest.raises(prep.PrepareRefused) as error:
            prep._real_inputs(args)
        assert type(error.value) is prep.PrepareRefused
        assert str(error.value) == f"chooser {role} producer disagrees with registered recipe"
        assert error.value.__cause__ is recipe_error
    assert prep.main(argv) == (2 if failure else 0)
    assert calls == [role] * (2 if failure == "recipe" else 1)
    if failure:
        diagnostic = ("unsupported chooser serving output" if failure == "output"
                      else f"chooser {role} producer disagrees with registered recipe")
        assert capsys.readouterr().err == f"refused: PrepareRefused: {diagnostic}\n"
        assert not out.exists()
        return
    assert capsys.readouterr().err == ""
    _assert_published_descriptor(out, role, cache, floor=floor)


@pytest.mark.parametrize("role", OUTPUTS)
def test_pinned_plan_only_refuses_chooser_panel_mismatch_before_plan(tmp_path, role, monkeypatch, capsys):
    pinned, _store, _snapshot, _records, _data = build_pinned(tmp_path, ALL_ONE)
    cache = _cache(role)
    monkeypatch.setattr(prep, "_real_inputs", lambda *a: (*_inputs(role, cache), _states(role, cache)))
    out = tmp_path / "out"
    assert prep.main(["--release-id", "r1", "--out", str(out), "--plan-only",
                      "--panel-catalog", str(pinned.catalog_path),
                      "--panel-snapshot-id", pinned.snapshot_id,
                      "--panel-artifact-root", str(pinned.artifact_root)]) == 2
    assert capsys.readouterr().err == "refused: PrepareRefused: chooser fold policy does not match the pinned panel\n"
    assert not out.exists()


@pytest.mark.parametrize("role", ["size", "unsupported", "runup_move"])
def test_explicit_policy_mapping_cannot_relabel_role(role):
    policy = _policy("implied_t1", _cache("implied_t1"))
    expected_type = prep.PrepareRefused if role == "runup_move" else ValueError
    with pytest.raises(expected_type) as error:
        prep.build_states({}, chooser_fold_policies={role: policy})
    assert type(error.value) is expected_type
    assert str(error.value) == ("chooser fold policy role disagrees with member" if role == "runup_move"
                                else "unsupported chooser fold role")


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("kind", ["none", "mapping", "duck-policy", "size-policy"])
def test_policy_mapping_requires_exact_chooser_policy_type(tmp_path, role, kind):
    cache = _cache(role)
    fields = dataclasses.asdict(_policy(role, cache))
    invalid = {"none": None, "mapping": fields, "duck-policy": SimpleNamespace(**fields),
               "size-policy": SizeFoldPolicy(model_id=cache["model_id"], feature_order=("x", "y"),
                                             panel_sha256="a" * 64, interval_floor=0.0)}[kind]
    with pytest.raises(prep.PrepareRefused) as error:
        prep.build_states({f"tier4_folds:{role}": {_name(cache): _bytes(cache)}},
                          chooser_fold_policies={role: invalid})
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold policy role disagrees with member"
    assert error.value.__cause__ is None
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("role", OUTPUTS)
@pytest.mark.parametrize("defect", ["policy-type", "member", "duplicate", "bytearray", "memoryview"])
def test_direct_state_guards_refuse_before_any_publication(tmp_path, role, defect):
    cache = _cache(role)
    states = _states(role, cache)
    build = next(row for row in states if row.spec.member_id == f"tier4_folds:{role}")
    if defect in ("policy-type", "member"):
        # Empty state isolates the boundary guard from deeper cache/binding validation.
        build.payloads.clear()
        if defect == "policy-type":
            build.chooser_fold_policy = SimpleNamespace(**dataclasses.asdict(build.chooser_fold_policy))
        else:
            other_role = next(other for other in OUTPUTS if other != role)
            build.spec = next(row.spec for row in states if row.spec.member_id == f"tier4_folds:{other_role}")
    elif defect == "duplicate":
        states.append(dataclasses.replace(build, payloads=dict(build.payloads)))
    else:
        convert = bytearray if defect == "bytearray" else memoryview
        build.payloads = {_name(cache): convert(_bytes(cache))}
    out = tmp_path / "out"
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(out, *_inputs(role, cache), states)
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == ("chooser fold policy role disagrees with member"
                                if defect in ("policy-type", "member")
                                else "duplicate or non-byte chooser fold object")
    assert error.value.__cause__ is None
    assert not out.exists()
    assert deployment.current_pointer(out / "deployment") is None


def test_existing_size_cli_case_with_synthetic_registered_producer(tmp_path, monkeypatch):
    from engine.data.features import tier4
    from tests.test_phase5_serving_folds import test_cli_gets_explicit_registered_policy as original

    cache = _cache("size")
    monkeypatch.setattr(tier4, "size_feature_model", lambda: SimpleNamespace(
        model_id=cache["model_id"], features=("x", "y"), interval_floor=0.0, produces="pred_abs_move"))

    def never(*args, **kwargs):
        pytest.fail("registry or live data read in synthetic CLI compatibility test")

    monkeypatch.setattr(tier4.registry_mod, "load_registry", never)
    monkeypatch.setattr(tier4.store, "read_table", never)
    monkeypatch.setattr(tier4.store, "file_sha256", never)
    original(tmp_path, cache, monkeypatch)


@pytest.mark.parametrize("role", OUTPUTS)
def test_ordinary_destination_substitution_refuses_before_catalog(tmp_path, role, monkeypatch):
    cache, real = _cache(role), prep.write_object

    def wrong_destination(root, payload):
        digest, rel = real(root, payload)
        (layout.deployment_root(root) / rel).write_bytes(b"substituted")
        return digest, rel

    monkeypatch.setattr(prep, "write_object", wrong_destination)
    with pytest.raises(prep.PrepareRefused) as error:
        prep.write_release(tmp_path, *_inputs(role, cache), _states(role, cache))
    assert type(error.value) is prep.PrepareRefused
    assert str(error.value) == "chooser fold object bytes differ from preflight"
    assert not (tmp_path / "phase5_release.json").exists()
    assert deployment.current_pointer(tmp_path / "deployment") is None


@pytest.mark.parametrize("role", OUTPUTS)
def test_ordinary_plan_only_refuses_malformed_cache_before_plan(tmp_path, role, monkeypatch, capsys):
    cache = _cache(role)
    cache["pool_res"] = [float("nan"), 0.0]
    monkeypatch.setattr(prep, "_real_inputs", lambda *a: (*_inputs(role, cache), _states(role, cache)))
    out = tmp_path / "out"
    assert prep.main(["--release-id", "r1", "--out", str(out), "--plan-only"]) == 2
    assert capsys.readouterr().err == (
        "refused: PrepareRefused: chooser fold refused: invalid serving fold held-out pool\n")
    assert not out.exists()


@pytest.mark.parametrize("role", OUTPUTS)
def test_ordinary_authoring_snapshots_caller_payloads_before_staging(tmp_path, role, monkeypatch):
    cache = _cache(role)
    states, real = _states(role, cache), deployment.stage_release
    original = next(build for build in states if build.spec.member_id == f"tier4_folds:{role}")

    def mutate_caller(*args, **kwargs):
        original.payloads.clear()
        return real(*args, **kwargs)

    monkeypatch.setattr(deployment, "stage_release", mutate_caller)
    prep.write_release(tmp_path, *_inputs(role, cache), states)
    assert original.payloads == {}
    _assert_published_descriptor(tmp_path, role, cache)
