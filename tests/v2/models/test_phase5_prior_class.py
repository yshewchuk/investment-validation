"""classify_prior: read-only prior classification for the pool-release bootstrap (#1102)."""
from __future__ import annotations

import json

import pytest

from checks import phase5_release_assembly as assembly
from engine.v2.foundation import content_hash
from engine.v2.models import deployment
from engine.v2.models import release_layout as phase5_release

from tests.v2.models.test_phase5_release_assembly import _stage_prior


def _cls(root, prior_release_id):
    return assembly.classify_prior(root, prior_release_id)


def test_absent_prior_id_is_none(tmp_path):
    cls = _cls(tmp_path, None)
    assert cls.kind == assembly.PRIOR_NONE
    assert cls.reason == "no prior release was requested"


def test_missing_pools_catalog_is_damaged(tmp_path):
    cls = _cls(tmp_path, "pools-2026-01-01")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "a pools-* release always has a catalog"


def test_missing_other_catalog_is_no_pool_catalog(tmp_path):
    cls = _cls(tmp_path, "someone-else")
    assert cls.kind == assembly.PRIOR_NO_POOL_CATALOG
    assert cls.reason == "the prior release has no catalog"


def test_unsafe_prior_id_is_damaged(tmp_path):
    cls = _cls(tmp_path, "../evil")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog could not be read or validated"


def test_catalog_without_pool_cutoff_is_no_pool_catalog(tmp_path):
    _stage_prior(tmp_path, prior_sources={"panel_sha": "x"})
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_NO_POOL_CATALOG
    assert cls.reason == "the prior catalog has no pool_cutoff"


def test_catalog_with_pool_cutoff_is_usable(tmp_path):
    _stage_prior(tmp_path)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_USABLE
    assert cls.reason == "the prior catalog has a pool_cutoff"


def test_symlinked_catalog_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    target = tmp_path / "elsewhere.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog could not be read or validated"


def test_oversize_catalog_is_damaged(tmp_path, monkeypatch):
    _stage_prior(tmp_path)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    monkeypatch.setattr(assembly, "MAX_CATALOG_BYTES", path.stat().st_size - 1)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog could not be read or validated"


def test_unparseable_catalog_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").write_bytes(b"{not json")
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog could not be read or validated"


def test_wrong_release_id_is_damaged(tmp_path):
    _, body = _stage_prior(tmp_path)
    phase5_release.write_catalog(phase5_release.catalog_path(tmp_path, "prior1"),
                                 {**body, "release_id": "other"})
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog could not be read or validated"


def test_pool_cutoff_not_a_date_is_damaged(tmp_path):
    _stage_prior(tmp_path, prior_sources={"pool_cutoff": "not-a-date"})
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def _rewrite(path, mutate):
    body = json.loads(path.read_bytes())
    body.pop("manifest_hash")
    mutate(body)
    body["manifest_hash"] = content_hash(body)
    path.write_text(json.dumps(body))


def test_empty_members_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"),
             lambda body: body.__setitem__("members", []))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_sources_not_a_dict_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"),
             lambda body: body.__setitem__("sources", "not-a-mapping"))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_malformed_object_row_is_damaged(tmp_path):
    _stage_prior(tmp_path)

    def corrupt_pool_row(body):
        row = next(r for r in body["members"] if r["member_id"] == "paired_residual_pool")
        row["objects"][0]["content_hash"] = "bad"

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), corrupt_pool_row)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_traversal_object_hash_is_damaged(tmp_path, monkeypatch):
    _stage_prior(tmp_path)

    def corrupt_pool_row(body):
        row = next(r for r in body["members"] if r["member_id"] == "paired_residual_pool")
        row["objects"][0]["content_hash"] = "sha256:../../../etc/passwd"

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), corrupt_pool_row)

    real_read = assembly._read_regular
    catalog_name = phase5_release.catalog_path(tmp_path, "prior1").name

    def guarded_read(path, *args, **kwargs):
        if str(path).endswith(catalog_name):
            return real_read(path, *args, **kwargs)
        raise AssertionError("object read before hash shape check")

    monkeypatch.setattr(assembly, "_read_regular", guarded_read)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_tampered_staged_manifest_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    manifest = phase5_release.staged_manifest_path(
        phase5_release.deployment_root(tmp_path), "prior1")
    body = json.loads(manifest.read_bytes())
    body["release"]["deployment_id"] = "x"
    manifest.write_text(json.dumps(body))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog disagrees with its staged release"


def test_corrupt_untouched_prior_object_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    catalog = json.loads(phase5_release.catalog_path(tmp_path, "prior1").read_bytes())
    row = next(r for r in catalog["members"] if r["member_id"] == "payoff_line:STR-THRU")
    obj = row["objects"][0]
    (phase5_release.deployment_root(tmp_path) / obj["path"]).write_bytes(b"tampered bytes")
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_tampered_pool_object_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    catalog = json.loads(phase5_release.catalog_path(tmp_path, "prior1").read_bytes())
    row = next(r for r in catalog["members"] if r["member_id"] == "paired_residual_pool")
    obj = row["objects"][0]
    path = phase5_release.deployment_root(tmp_path) / obj["path"]
    path.write_bytes(b"x" * len(path.read_bytes()))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_negative_object_bytes_is_damaged(tmp_path):
    _stage_prior(tmp_path)

    def corrupt_pool_row(body):
        row = next(r for r in body["members"] if r["member_id"] == "paired_residual_pool")
        row["objects"][0]["bytes"] = -1

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), corrupt_pool_row)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


@pytest.mark.parametrize("value", [True, 5.0, -1])
def test_untouched_row_bool_or_float_bytes_is_damaged(tmp_path, value):
    _stage_prior(tmp_path)

    def corrupt_untouched_row(body):
        row = next(r for r in body["members"] if r["member_id"] == "payoff_line:STR-THRU")
        row["objects"][0]["bytes"] = value

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), corrupt_untouched_row)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


@pytest.mark.parametrize("value", [True, 2.5])
def test_non_int_pool_object_bytes_is_damaged(tmp_path, value):
    _stage_prior(tmp_path)

    def corrupt_pool_row(body):
        row = next(r for r in body["members"] if r["member_id"] == "paired_residual_pool")
        row["objects"][0]["bytes"] = value

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), corrupt_pool_row)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog is malformed"


def test_legacy_catalog_without_pool_rows_is_no_pool_catalog(tmp_path):
    _stage_prior(tmp_path, prior_sources={"panel_sha": "x"})

    def keep_only_untouched_row(body):
        body["members"] = [row for row in body["members"]
                           if row["member_id"] == "payoff_line:STR-THRU"]

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), keep_only_untouched_row)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_NO_POOL_CATALOG
    assert cls.reason == "the prior catalog has no pool_cutoff"


def test_corrupt_model_binding_payload_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    store = phase5_release.deployment_root(tmp_path)
    staged = deployment.resolve_release(store, "prior1")
    member_path = store / staged.bindings[0].members[0].path
    member_path.write_bytes(b"x" * len(member_path.read_bytes()))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    # stage 3c: verify_staged only checks the staged manifest, not payload
    # bytes, so the corruption reaches the pool_cutoff block's
    # _read_prior_payloads read ("a prior model object is corrupt") and
    # classify_prior reports "the prior catalog is malformed".
    assert cls.reason == "the prior catalog is malformed"


def test_deployment_id_mismatch_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"),
             lambda body: body.__setitem__("deployment_id", "other-deployment"))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog disagrees with its staged release"


def test_no_pool_cutoff_catalog_with_wrong_deployment_id_is_damaged(tmp_path):
    _stage_prior(tmp_path, prior_sources={"panel_sha": "x"})
    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"),
             lambda body: body.__setitem__("deployment_id", "other-deployment"))
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason == "the prior catalog disagrees with its staged release"


def _snapshot(root):
    snapshot = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            snapshot[str(p.relative_to(root)) + "/"] = b"<dir>"
        else:
            snapshot[str(p.relative_to(root))] = p.read_bytes()
    return snapshot


def test_classify_prior_writes_nothing(tmp_path):
    _stage_prior(tmp_path)
    before = _snapshot(tmp_path)
    for prior_id in ("prior1", None, "absent", "pools-2026-01-01", "../evil"):
        assembly.classify_prior(tmp_path, prior_id)
    assert _snapshot(tmp_path) == before
