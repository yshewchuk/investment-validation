"""S4A: the executor's staged refresh identity is a pure function of the claim.

The staged document must be reproducible from the admitted, hashed JobSpec
alone -- never the environment and never a legacy path -- because it decides
which catalog and objects root a supervised refresh attempt commits into.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from engine.v2.foundation import to_document
from engine.v2.ops.incremental_data import RefreshParameters
from engine.v2.ops.refresh_staging import (
    REFRESH_INPUT_DOCUMENT_NAMES,
    stage_refresh_input,
)


def _parameters():
    return RefreshParameters(
        expected_ids=("request-1",), parent_snapshot_id="snap-parent",
        refresh_plan_hash="sha256:" + "a" * 64, provider_calls=0,
        catalog_path="/ops/catalog.sqlite", objects_root="/ops/objects",
        scope="shadow", expected_head_generation=3,
        expected_head_snapshot_id="snap-parent", table_name="daily_market")


def _claim(parameters):
    return SimpleNamespace(spec=SimpleNamespace(
        kind="incremental_refresh", parameters=to_document(parameters)))


def test_staged_document_is_byte_identical_across_calls_and_environment(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    staging.mkdir()
    claim = _claim(_parameters())
    name = REFRESH_INPUT_DOCUMENT_NAMES["incremental_refresh"]

    stage_refresh_input(claim, staging)
    first = (staging / name).read_bytes()

    monkeypatch.setenv("INVESTING_PLAN_ROOT", str(tmp_path / "unrelated"))
    monkeypatch.setenv("RAW_FETCH", str(tmp_path / "legacy-fetch"))
    stage_refresh_input(claim, staging)
    second = (staging / name).read_bytes()

    assert first == second
    assert json.loads(second) == {
        "catalog_path": "/ops/catalog.sqlite", "objects_root": "/ops/objects",
        "scope": "shadow", "expected_head_generation": 3,
        "expected_head_snapshot_id": "snap-parent", "table_name": "daily_market",
        "fetch_root": str(staging / "fetch"),
    }


def test_kind_without_a_registered_document_stages_nothing(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    claim = SimpleNamespace(spec=SimpleNamespace(kind="incremental_backfill", parameters={}))
    stage_refresh_input(claim, staging)
    assert list(staging.iterdir()) == []
