from __future__ import annotations

import pytest

from engine.v2.models import release_layout
from engine.v2.scoring import release_bindings


@pytest.mark.parametrize("sources", [
    {"pool_cutoff": "2026-11-10", "retrain_cutoff": "2026-11-1"},
    {"pool_cutoff": "2026-11-10", "retrain_cutoff": "2026-11-11"},
])
def test_scoring_catalog_reader_rejects_invalid_retrain_cutoff(tmp_path, sources):
    payload = b"catalog-test-object"
    digest = release_layout.sha256_bytes(payload)
    object_row = {
        "name": "prior.bin",
        "path": release_layout.object_relpath(digest),
        "content_hash": digest,
        "bytes": len(payload),
    }
    rows = [release_layout.member_row(spec, release_layout.STAGED, [object_row], "prior")
            for spec in release_layout.STATE_SPECS]
    body = release_layout.manifest_body("base1", "deployment1", rows, sources)
    release_layout.write_catalog(release_layout.catalog_path(tmp_path, "base1"), body)

    with pytest.raises(release_bindings.ModelNotReady) as err:
        release_bindings._read_state_catalog(tmp_path, "base1")

    assert "cutoff metadata is invalid" in str(err.value)
