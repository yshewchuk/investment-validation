"""P5-4: ``_N_ADMISSIBLE_BY_DEPTH`` as a versioned, hashed, release-pinnable table.

Legacy (``engine/score.py``) is imported here, never by ``engine/v2``: these
tests are what keeps the v2 copy of the literal and the legacy literal equal,
and what makes any edit to either a visible identity change.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from engine.score import _N_ADMISSIBLE_MEDIAN, Scorer
from engine.v2.models.admissible_table import (
    N_ADMISSIBLE_BY_DEPTH_V1_HASH,
    AdmissibleTableError,
    legacy_n_admissible_table,
    make_admissible_depth_table,
    n_admissible_for,
)
from engine.v2.models.frozen_release import inventory_member, release_member
from engine.v2.models.frozen_state import (
    FrozenStateError,
    FrozenStateLoader,
    FrozenStateRef,
    serialize_frozen_state,
)
from engine.v2.models.lineage import DataDependency, Lineage


def _legacy_lookup(depth: float) -> float:
    shim = SimpleNamespace(_N_ADMISSIBLE_BY_DEPTH=Scorer._N_ADMISSIBLE_BY_DEPTH)
    return Scorer._n_admissible_for(shim, depth)


def test_v2_table_is_the_legacy_literal():
    table = legacy_n_admissible_table()
    assert table.breakpoints == tuple(Scorer._N_ADMISSIBLE_BY_DEPTH)
    assert table.fallback == _N_ADMISSIBLE_MEDIAN


def test_identity_is_pinned_and_reproducible():
    table = legacy_n_admissible_table()
    assert table.content_hash == N_ADMISSIBLE_BY_DEPTH_V1_HASH
    assert table.key == ("dyn_sv.n_admissible_by_depth", "v1")
    assert serialize_frozen_state(table) == serialize_frozen_state(legacy_n_admissible_table())


@pytest.mark.parametrize("depth", [
    float("nan"), float("inf"), -5.0, 0.0, 5.999, 6.0, 6.5, 10.0, 12.9, 13.0,
    14.0, 25.99, 26.0, 61.0, 62.0, 175.9, 176.0, 500.0,
])
def test_lookup_matches_legacy_n_admissible_for(depth):
    assert n_admissible_for(legacy_n_admissible_table(), depth) == _legacy_lookup(depth)


def test_any_edit_is_a_new_identity():
    base = legacy_n_admissible_table()
    points = list(base.breakpoints)
    points[4] = (points[4][0], points[4][1] + 0.5)
    edited = make_admissible_depth_table(
        table_id=base.table_id, version=base.version, breakpoints=points,
        fallback=base.fallback, provenance=base.provenance, lineage=base.lineage)
    assert edited.content_hash != base.content_hash


@pytest.mark.parametrize("points", [(), ((6.0, float("nan")),), ((7.0, 1.0), (6.0, 2.0))])
def test_malformed_tables_refuse(points):
    with pytest.raises(AdmissibleTableError):
        make_admissible_depth_table(
            table_id="t", version="v", breakpoints=points, fallback=1.0,
            provenance="p", lineage=Lineage(data=(DataDependency(table="x"),)))


def test_loader_and_release_members(tmp_path):
    table = legacy_n_admissible_table()
    (tmp_path / "table.json").write_bytes(serialize_frozen_state(table))
    ref = FrozenStateRef(path="table.json", content_hash=N_ADMISSIBLE_BY_DEPTH_V1_HASH)
    assert FrozenStateLoader(tmp_path).load(ref) == table
    assert release_member(table, "calibration").content_hash == N_ADMISSIBLE_BY_DEPTH_V1_HASH
    assert inventory_member(table, "chooser:n_admissible", "artifact://t").kind == "calibration"
    (tmp_path / "table.json").write_bytes(
        serialize_frozen_state(table).replace(b"31.5", b"31.6"))
    with pytest.raises(FrozenStateError):
        FrozenStateLoader(tmp_path).load(ref)
