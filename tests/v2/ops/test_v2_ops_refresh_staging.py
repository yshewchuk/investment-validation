"""S4A: the executor's staged refresh identity is a pure function of the claim.

The staged document must be reproducible from the admitted, hashed JobSpec
alone -- never the environment and never a legacy path -- because it decides
which catalog and objects root a supervised refresh attempt commits into.

S4C Part 3 adds ``computed_moves_refresh``'s own staged document; the S4C
follow-up gives ``forward_calendar_refresh`` its own much smaller one --
``attempt_id``/``fence`` only (see ``refresh_staging.py``'s module docstring).
``artifact_check`` is now this module's example of "a kind without a
registered document".
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from engine.v2.foundation import to_document
from engine.v2.ops.calendar_moves_jobs import CalendarMovesParameters
from engine.v2.ops.incremental_data import RefreshParameters
from engine.v2.ops.refresh_staging import stage_refresh_input
from engine.v2.ops.stores.refresh_contracts import REFRESH_INPUT_DOCUMENT_NAMES


def _parameters():
    return RefreshParameters(
        expected_ids=("request-1",), parent_snapshot_id="snap-parent",
        refresh_plan_hash="sha256:" + "a" * 64, provider_calls=0,
        catalog_path="/ops/catalog.sqlite", objects_root="/ops/objects",
        scope="shadow", expected_head_generation=3,
        expected_head_snapshot_id="snap-parent", table_name="daily_market")


def _claim(parameters, *, kind="incremental_refresh", attempt_id="att-1", fence=1):
    return SimpleNamespace(spec=SimpleNamespace(
        kind=kind, parameters=to_document(parameters)),
        attempt_id=attempt_id, fence=fence)


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
    claim = SimpleNamespace(spec=SimpleNamespace(kind="artifact_check", parameters={}))
    stage_refresh_input(claim, staging)
    assert list(staging.iterdir()) == []


def _computed_moves_parameters():
    return CalendarMovesParameters(
        expected_ids=("AAPL",), parent_snapshot_id="snap-parent",
        parent_receipt_id="receipt-parent",
        refresh_plan_hash="sha256:" + "b" * 64, provider_calls=0,
        catalog_path="/ops/catalog.sqlite", objects_root="/ops/objects",
        scope="shadow", expected_head_generation=3,
        expected_head_snapshot_id="snap-parent", as_of="2026-09-27",
        all_scoreable=True, since=None)


def test_computed_moves_refresh_stages_its_own_document_with_attempt_and_fence(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    claim = _claim(_computed_moves_parameters(), kind="computed_moves_refresh",
                  attempt_id="att-cm-1", fence=2)
    name = REFRESH_INPUT_DOCUMENT_NAMES["computed_moves_refresh"]

    stage_refresh_input(claim, staging)

    assert json.loads((staging / name).read_bytes()) == {
        "catalog_path": "/ops/catalog.sqlite", "objects_root": "/ops/objects",
        "scope": "shadow", "expected_head_generation": 3,
        "expected_head_snapshot_id": "snap-parent",
        "parent_receipt_id": "receipt-parent", "as_of": "2026-09-27",
        "all_scoreable": True, "since": None,
        "attempt_id": "att-cm-1", "fence": 2,
    }


def test_computed_moves_refresh_document_is_byte_identical_across_calls(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    claim = _claim(_computed_moves_parameters(), kind="computed_moves_refresh")
    name = REFRESH_INPUT_DOCUMENT_NAMES["computed_moves_refresh"]

    stage_refresh_input(claim, staging)
    first = (staging / name).read_bytes()
    stage_refresh_input(claim, staging)
    second = (staging / name).read_bytes()

    assert first == second


def test_forward_calendar_refresh_stages_only_its_attempt_and_fence(tmp_path):
    staging = tmp_path / "staging"
    staging.mkdir()
    claim = _claim(_computed_moves_parameters(), kind="forward_calendar_refresh",
                   attempt_id="att-fc-1", fence=4)
    name = REFRESH_INPUT_DOCUMENT_NAMES["forward_calendar_refresh"]

    stage_refresh_input(claim, staging)

    assert json.loads((staging / name).read_bytes()) == {
        "attempt_id": "att-fc-1", "fence": 4,
    }
