"""classify_prior: read-only prior classification for the pool-release bootstrap (#1102)."""
from __future__ import annotations

import json

from checks import phase5_release_assembly as assembly
from engine.v2.foundation import content_hash
from engine.v2.models import release_layout as phase5_release

from tests.v2.models.test_phase5_release_assembly import _stage_prior


def _kind(root, prior_release_id):
    return assembly.classify_prior(root, prior_release_id).kind


def test_absent_prior_id_is_none(tmp_path):
    assert _kind(tmp_path, None) == assembly.PRIOR_NONE


def test_missing_pools_catalog_is_damaged(tmp_path):
    assert _kind(tmp_path, "pools-2026-01-01") == assembly.PRIOR_DAMAGED


def test_missing_other_catalog_is_no_pool_catalog(tmp_path):
    assert _kind(tmp_path, "someone-else") == assembly.PRIOR_NO_POOL_CATALOG


def test_unsafe_prior_id_is_damaged(tmp_path):
    assert _kind(tmp_path, "../evil") == assembly.PRIOR_DAMAGED


def test_catalog_without_pool_cutoff_is_no_pool_catalog(tmp_path):
    _stage_prior(tmp_path, prior_sources={"panel_sha": "x"})
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_NO_POOL_CATALOG


def test_catalog_with_pool_cutoff_is_usable(tmp_path):
    _stage_prior(tmp_path)
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_USABLE


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
    assert _kind(tmp_path, "prior1") == assembly.PRIOR_DAMAGED


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


def test_classify_prior_writes_nothing(tmp_path):
    _stage_prior(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    for prior_id in ("prior1", None, "absent", "pools-2026-01-01", "../evil"):
        assembly.classify_prior(tmp_path, prior_id)
    after = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    assert after == before
