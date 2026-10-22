"""classify_prior: read-only prior classification for the pool-release bootstrap (#1102)."""
from __future__ import annotations

import json

from checks import phase5_release_assembly as assembly
from engine.v2.foundation import content_hash
from engine.v2.models import release_layout as phase5_release

from tests.v2.models.test_phase5_release_assembly import _stage_prior


def _cls(root, prior_release_id):
    return assembly.classify_prior(root, prior_release_id)


def _kind(root, prior_release_id):
    return _cls(root, prior_release_id).kind


_REASONS = {
    assembly.PRIOR_NONE: {"no prior release was requested"},
    assembly.PRIOR_NO_POOL_CATALOG: {
        "the prior release has no catalog",
        "the prior catalog has no pool_cutoff",
    },
    assembly.PRIOR_USABLE: {"the prior catalog has a pool_cutoff"},
    assembly.PRIOR_DAMAGED: {
        "the prior catalog could not be read or validated",
        "the prior catalog disagrees with its staged release",
        "the prior catalog is malformed",
        "a pools-* release always has a catalog",
    },
}


def test_absent_prior_id_is_none(tmp_path):
    cls = _cls(tmp_path, None)
    assert cls.kind == assembly.PRIOR_NONE
    assert cls.reason in _REASONS[assembly.PRIOR_NONE]


def test_missing_pools_catalog_is_damaged(tmp_path):
    cls = _cls(tmp_path, "pools-2026-01-01")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason in _REASONS[assembly.PRIOR_DAMAGED]


def test_missing_other_catalog_is_no_pool_catalog(tmp_path):
    cls = _cls(tmp_path, "someone-else")
    assert cls.kind == assembly.PRIOR_NO_POOL_CATALOG
    assert cls.reason in _REASONS[assembly.PRIOR_NO_POOL_CATALOG]


def test_unsafe_prior_id_is_damaged(tmp_path):
    assert _kind(tmp_path, "../evil") == assembly.PRIOR_DAMAGED


def test_catalog_without_pool_cutoff_is_no_pool_catalog(tmp_path):
    _stage_prior(tmp_path, prior_sources={"panel_sha": "x"})
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_NO_POOL_CATALOG
    assert cls.reason in _REASONS[assembly.PRIOR_NO_POOL_CATALOG]


def test_catalog_with_pool_cutoff_is_usable(tmp_path):
    _stage_prior(tmp_path)
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_USABLE
    assert cls.reason in _REASONS[assembly.PRIOR_USABLE]


def test_symlinked_catalog_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    target = tmp_path / "elsewhere.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def test_oversize_catalog_is_damaged(tmp_path, monkeypatch):
    _stage_prior(tmp_path)
    path = phase5_release.catalog_path(tmp_path, "prior1")
    monkeypatch.setattr(assembly, "MAX_CATALOG_BYTES", path.stat().st_size - 1)
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def test_unparseable_catalog_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    phase5_release.catalog_path(tmp_path, "prior1").write_bytes(b"{not json")
    cls = _cls(tmp_path, "prior1")
    assert cls.kind == assembly.PRIOR_DAMAGED
    assert cls.reason in _REASONS[assembly.PRIOR_DAMAGED]


def test_wrong_release_id_is_damaged(tmp_path):
    _, body = _stage_prior(tmp_path)
    phase5_release.write_catalog(phase5_release.catalog_path(tmp_path, "prior1"),
                                 {**body, "release_id": "other"})
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def test_pool_cutoff_not_a_date_is_damaged(tmp_path):
    _stage_prior(tmp_path, prior_sources={"pool_cutoff": "not-a-date"})
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


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
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def test_sources_not_a_dict_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"),
             lambda body: body.__setitem__("sources", "not-a-mapping"))
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def test_malformed_object_row_is_damaged(tmp_path):
    _stage_prior(tmp_path)

    def corrupt_pool_row(body):
        row = next(r for r in body["members"] if r["member_id"] == "paired_residual_pool")
        row["objects"][0]["content_hash"] = "bad"

    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"), corrupt_pool_row)
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def test_deployment_id_mismatch_is_damaged(tmp_path):
    _stage_prior(tmp_path)
    _rewrite(phase5_release.catalog_path(tmp_path, "prior1"),
             lambda body: body.__setitem__("deployment_id", "other-deployment"))
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


def _snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_classify_prior_writes_nothing(tmp_path):
    _stage_prior(tmp_path)
    before = _snapshot(tmp_path)
    for prior_id in ("prior1", None, "absent", "pools-2026-01-01", "../evil"):
        assembly.classify_prior(tmp_path, prior_id)
    assert _snapshot(tmp_path) == before
